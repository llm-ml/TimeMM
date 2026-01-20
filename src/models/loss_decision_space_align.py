# decision_space_align_loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


AlignMode = Literal["pairwise_sym", "teacher_student", "to_mixture"]


@dataclass
class DecisionAlignCfg:
    mode: AlignMode = "teacher_student"
    temperature: float = 1.0          # T in sigmoid(m/T)
    teacher_index: int = -1           # for teacher_student: default last scale is teacher
    detach_teacher: bool = True       # teacher prob detached by default
    symmetrize: bool = True           # used in pairwise_sym
    reduction: Literal["mean", "sum"] = "mean"
    eps: float = 1e-6                 # for numerical safety


class DecisionSpaceAlignLoss(nn.Module):
    """
    Decision-space alignment across K time scales
    """

    def __init__(self, cfg: Optional[DecisionAlignCfg] = None):
        super().__init__()
        self.cfg = cfg or DecisionAlignCfg()

    @staticmethod
    def _margins(
        user_rep_k: List[torch.Tensor],
        pos_item_rep_k: List[torch.Tensor],
        neg_item_rep_k: List[torch.Tensor],
    ) -> torch.Tensor:
        # returns logits: [K, B]
        K = len(user_rep_k)
        assert K == len(pos_item_rep_k) == len(neg_item_rep_k), "K mismatch in inputs"
        logits = []
        for u, ip, ineg in zip(user_rep_k, pos_item_rep_k, neg_item_rep_k):
            # [B]
            pos = (u * ip).sum(dim=-1)
            neg = (u * ineg).sum(dim=-1)
            logits.append(pos - neg)
        return torch.stack(logits, dim=0)

    def forward(
        self,
        user_rep_k: List[torch.Tensor],
        pos_item_rep_k: List[torch.Tensor],
        neg_item_rep_k: List[torch.Tensor],
        gate_w: Optional[torch.Tensor] = None,  # [B, K], optional for to_mixture
    ) -> torch.Tensor:
        cfg = self.cfg
        K = len(user_rep_k)
        if K <= 1:
            # nothing to align
            return user_rep_k[0].new_zeros(())

        logits = self._margins(user_rep_k, pos_item_rep_k, neg_item_rep_k)  # [K,B]
        T = float(max(cfg.temperature, 1e-8))
        logits_T = logits / T  # [K,B]
        probs = torch.sigmoid(logits_T)  # [K,B]

        if cfg.mode == "teacher_student":
            t_idx = cfg.teacher_index if cfg.teacher_index >= 0 else (K + cfg.teacher_index)
            t_idx = int(max(0, min(K - 1, t_idx)))

            p_t = probs[t_idx]  # [B]
            if cfg.detach_teacher:
                p_t = p_t.detach()

            losses = []
            for k in range(K):
                if k == t_idx:
                    continue
                # distill teacher prob into student logits
                losses.append(F.binary_cross_entropy_with_logits(logits_T[k], p_t, reduction="none"))  # [B]
            loss = torch.stack(losses, dim=0).mean(dim=0)  # [B]
            return loss.mean() if cfg.reduction == "mean" else loss.sum()

        elif cfg.mode == "pairwise_sym":
            losses = []
            for i in range(K):
                for j in range(i + 1, K):
                    p_j = probs[j].detach() if cfg.detach_teacher else probs[j]
                    li = F.binary_cross_entropy_with_logits(logits_T[i], p_j, reduction="none")  # [B]
                    if cfg.symmetrize:
                        p_i = probs[i].detach() if cfg.detach_teacher else probs[i]
                        lj = F.binary_cross_entropy_with_logits(logits_T[j], p_i, reduction="none")  # [B]
                        lij = 0.5 * (li + lj)
                    else:
                        lij = li
                    losses.append(lij)
            loss = torch.stack(losses, dim=0).mean(dim=0)  # [B]
            return loss.mean() if cfg.reduction == "mean" else loss.sum()

        elif cfg.mode == "to_mixture":
            if gate_w is None:
                raise ValueError("mode='to_mixture' requires gate_w of shape [B,K].")
            if gate_w.dim() != 2 or gate_w.size(0) != logits.size(1) or gate_w.size(1) != K:
                raise ValueError(f"gate_w must be [B,K]={logits.size(1),K}, got {tuple(gate_w.shape)}")

            # mixture prob in probability space (safer than mixing logits)
            p_mix = (gate_w.transpose(0, 1) * probs).sum(dim=0)  # [B]
            p_mix = p_mix.clamp(min=cfg.eps, max=1.0 - cfg.eps)
            if cfg.detach_teacher:
                p_mix = p_mix.detach()

            # each scale aligns to mixture's decision prob
            per_k = []
            for k in range(K):
                per_k.append(F.binary_cross_entropy_with_logits(logits_T[k], p_mix, reduction="none"))  # [B]
            loss = torch.stack(per_k, dim=0).mean(dim=0)  # [B]
            return loss.mean() if cfg.reduction == "mean" else loss.sum()

        else:
            raise ValueError(f"Unknown cfg.mode={cfg.mode}")

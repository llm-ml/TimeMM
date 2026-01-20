# loss_multiscale_complement.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiScaleCompCfg:
    # off-diagonal correlation penalty strength is applied
    eps: float = 1e-8

    # optional standardize per-scale margins
    use_corr: bool = True  # True: correlation-based, False: covariance-based

    # encourage each scale to have non-trivial margin variance, prevents collapse
    use_var_floor: bool = True
    var_floor: float = 1e-3   # target minimum std of margins (after centering)
    var_weight: float = 0.1   # relative weight for var-floor term


class MultiScaleComplementLoss(nn.Module):
    """
    Complementarity loss across K time scales in decision space.
    """
    def __init__(self, cfg: Optional[MultiScaleCompCfg] = None):
        super().__init__()
        self.cfg = cfg or MultiScaleCompCfg()

    @staticmethod
    def _stack_list(x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        list of K tensors
        """
        if isinstance(x, list):
            assert len(x) > 0
            return torch.stack(x, dim=0)
        assert x.dim() == 3, f"Expected [K,B,D], got {tuple(x.shape)}"
        return x

    def forward(
        self,
        user_rep_k: Union[List[torch.Tensor], torch.Tensor],      # list([B,D]) or [K,B,D]
        pos_item_rep_k: Union[List[torch.Tensor], torch.Tensor],  # same
        neg_item_rep_k: Union[List[torch.Tensor], torch.Tensor],  # same
    ) -> torch.Tensor:
        U = self._stack_list(user_rep_k)        # [K,B,D]
        P = self._stack_list(pos_item_rep_k)    # [K,B,D]
        N = self._stack_list(neg_item_rep_k)    # [K,B,D]
        assert U.shape == P.shape == N.shape, f"Shape mismatch: U={U.shape}, P={P.shape}, N={N.shape}"

        K, B, D = U.shape
        if K <= 1:
            # no complementarity to enforce
            return U.new_zeros(())

        # margins in decision space
        pos_score = (U * P).sum(dim=-1)  # [K,B]
        neg_score = (U * N).sum(dim=-1)  # [K,B]
        M = pos_score - neg_score        # [K,B]
        M = M - M.mean(dim=1, keepdim=True)  # [K,B]

        # covariance correlation matrix across scales: [K,K]
        # cov = (M @ M^T) / B
        cov = (M @ M.transpose(0, 1)) / max(B, 1)

        if self.cfg.use_corr:
            # corr = cov / (std_i * std_j)
            var = torch.diag(cov).clamp_min(self.cfg.eps)  # [K]
            std = torch.sqrt(var)                          # [K]
            denom = (std[:, None] * std[None, :]).clamp_min(self.cfg.eps)
            C = cov / denom
        else:
            C = cov

        # off-diagonal squared penalty
        off = C - torch.diag(torch.diag(C))
        loss_comp = (off ** 2).mean()  # normalized, stable across K

        # optional avoid degenerate constant margins
        if self.cfg.use_var_floor:
            # std of centered margins per scale
            std_m = torch.sqrt((M * M).mean(dim=1) + self.cfg.eps)  # [K]
            # penalize if std below floor
            loss_var = F.relu(self.cfg.var_floor - std_m).mean()
            loss = loss_comp + self.cfg.var_weight * loss_var
        else:
            loss = loss_comp

        return loss

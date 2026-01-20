# multiscale_order_smoothness_loss.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OrderSmoothnessCfg:
    # If scales are ordered from short-horizon -> long-horizon, keep True:
    # we enforce E_short >= E_long, i.e., E[k] >= E[k+1]
    short_to_long: bool = True

    # hinge margin: enforce E[k] >= E[k+1] + margin
    margin: float = 0.0
    normalize_by_dim: bool = True

    # reduction for final loss
    reduction: str = "mean"  # "mean" or "sum"


class MultiScaleOrderSmoothnessLoss(nn.Module):
    def __init__(self, cfg: Optional[OrderSmoothnessCfg] = None):
        super().__init__()
        self.cfg = cfg or OrderSmoothnessCfg()

    def forward(
        self,
        user_rep_k: List[torch.Tensor],
        pos_item_rep_k: List[torch.Tensor],
        return_energies: bool = False,              # batch-level [K]
        return_sample_energies: bool = False,       # sample-level [K,B]
        return_sample_violations: bool = False,     # sample-level [K-1,B]
    ):
        assert isinstance(user_rep_k, (list, tuple)) and isinstance(pos_item_rep_k, (list, tuple))
        K = len(user_rep_k)
        assert K == len(pos_item_rep_k) and K >= 2, f"K must match and >=2, got {K} vs {len(pos_item_rep_k)}"

        energies_mean = []     # list of scalar
        energies_each = []     # list of [B]

        for uk, ik in zip(user_rep_k, pos_item_rep_k):
            assert uk.shape == ik.shape and uk.dim() == 2, f"Expect [B,D], got {uk.shape} vs {ik.shape}"
            diff = uk - ik
            e = (diff * diff).sum(dim=1)  # [B]
            if self.cfg.normalize_by_dim:
                e = e / float(uk.size(1))
            energies_each.append(e)        # [B]
            energies_mean.append(e.mean()) # scalar

        # batch-level energies: [K]
        E = torch.stack(energies_mean, dim=0)  # [K]
        # sample-level energies: [K,B]
        E_each = torch.stack(energies_each, dim=0)  # [K,B]

        # If list is long->short, flip both
        if not self.cfg.short_to_long:
            E = torch.flip(E, dims=[0])
            E_each = torch.flip(E_each, dims=[0])

        # loss computed on batch-level energies
        rhs = E[1:] + float(self.cfg.margin)
        lhs = E[:-1]
        violations = F.relu(rhs - lhs)  # [K-1]

        if self.cfg.reduction == "sum":
            loss = violations.sum()
        else:
            loss = violations.mean()

        # optional sample-level violations
        sample_viol = None
        if return_sample_violations:
            rhs_each = E_each[1:, :] + float(self.cfg.margin)   # [K-1,B]
            lhs_each = E_each[:-1, :]                           # [K-1,B]
            sample_viol = F.relu(rhs_each - lhs_each)           # [K-1,B]

        # return pack, backward compatible
        if return_energies or return_sample_energies or return_sample_violations:
            outs = [loss]
            if return_energies:
                # energies in original order as inputs
                E_out = E if self.cfg.short_to_long else torch.flip(E, dims=[0])
                outs.append(E_out)
            if return_sample_energies:
                E_each_out = E_each if self.cfg.short_to_long else torch.flip(E_each, dims=[0])
                outs.append(E_each_out)
            if return_sample_violations:
                # sample_viol already aligned to short_to_long after flip
                sample_viol_out = sample_viol if self.cfg.short_to_long else torch.flip(sample_viol, dims=[0])
                outs.append(sample_viol_out)
            return tuple(outs) if len(outs) > 1 else outs[0]

        return loss

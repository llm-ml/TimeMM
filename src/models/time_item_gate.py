# time_item_gate.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class ItemTimeFeatConfig:
    """
    train-only item time features raw seconds / counts.
    """
    use_age: bool = True          # age = t_ref - t_last(item)
    use_span: bool = True         # span = t_last - t_first
    use_cnt: bool = True          # cnt_train
    use_mean_gap: bool = True     # mean gap between consecutive interactions for the item

    # global anchor for age
    # t_ref = max timestamp in train
    use_global_anchor: bool = True


class ItemTimeFeatBuilder:
    @staticmethod
    def build_from_train_df(
        df_train: pd.DataFrame,
        iid_field: str,
        time_field: str,
        num_item: int,
        cfg: ItemTimeFeatConfig = ItemTimeFeatConfig(),
    ) -> torch.Tensor:
        """
        Return item_time_feat: FloatTensor [I, F]
        Missing items -> all zeros. This uses TRAIN ONLY df. No leakage.
        """

        if iid_field not in df_train.columns:
            raise ValueError(f"iid_field='{iid_field}' not in df_train.columns")
        if time_field not in df_train.columns:
            raise ValueError(f"time_field='{time_field}' not in df_train.columns")

        # keep only needed cols
        df = df_train[[iid_field, time_field]].copy()
        df[time_field] = df[time_field].astype(np.int64)

        # global anchor
        if cfg.use_global_anchor:
            t_ref = int(df[time_field].max())
        else:
            # fallback: still use global to avoid per-item inconsistencies
            t_ref = int(df[time_field].max())

        g = df.groupby(iid_field, as_index=True)[time_field]
        t_last = g.max()
        t_first = g.min()
        cnt = g.size().astype(np.int64)

        # mean gap, sort within each item and diff timestamps
        df_sorted = df.sort_values([iid_field, time_field], kind="mergesort")
        gap = (
            df_sorted.groupby(iid_field)[time_field]
            .diff()
            .fillna(0)
            .clip(lower=0)
            .astype(np.int64)
        )
        df_sorted["_gap"] = gap
        mean_gap = df_sorted.groupby(iid_field)["_gap"].mean().astype(np.float32)

        # allocate arrays, zeros for missing items
        feats: List[np.ndarray] = []

        if cfg.use_age:
            age = (t_ref - t_last).astype(np.int64)
            age = np.maximum(age.to_numpy(), 0).astype(np.float32)
            feats.append(_scatter_to_len(age, t_last.index.to_numpy(), num_item))

        if cfg.use_span:
            span = (t_last - t_first).astype(np.int64)
            span = np.maximum(span.to_numpy(), 0).astype(np.float32)
            feats.append(_scatter_to_len(span, t_last.index.to_numpy(), num_item))

        if cfg.use_cnt:
            c = cnt.to_numpy().astype(np.float32)
            feats.append(_scatter_to_len(c, cnt.index.to_numpy(), num_item))

        if cfg.use_mean_gap:
            mg = mean_gap.to_numpy().astype(np.float32)
            feats.append(_scatter_to_len(mg, mean_gap.index.to_numpy(), num_item))

        if len(feats) == 0:
            raise ValueError("No item time features enabled in ItemTimeFeatConfig")

        # [I, F]
        mat = np.stack(feats, axis=1).astype(np.float32)
        return torch.from_numpy(mat)


def _scatter_to_len(values: np.ndarray, ids: np.ndarray, length: int) -> np.ndarray:
    """
    Scatter per-id values into a dense array of shape
    """
    out = np.zeros((length,), dtype=np.float32)
    # safety: ignore out-of-range
    mask = (ids >= 0) & (ids < length)
    out[ids[mask]] = values[mask]
    return out


class TimeAwareItemModalGate(nn.Module):
    """
    Item-side time-aware gate for 3 modalities
    """
    def __init__(self, in_dim: int, hidden: int = 32, alpha: float = 0.5, learn_base_logits: bool = True):
        super().__init__()
        self.alpha = float(alpha)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        if learn_base_logits:
            self.base_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        else:
            self.register_buffer("base_logits", torch.zeros(3, dtype=torch.float32))

    def forward(self, item_time_feat: torch.Tensor, present_modal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        delta = self.mlp(item_time_feat)  # [I,3]
        logits = self.base_logits.view(1, 3) + self.alpha * delta

        if present_modal_mask is not None:
            # mask absent modality by setting very negative logits
            m = present_modal_mask.view(1, 3).to(dtype=logits.dtype, device=logits.device)
            logits = logits + (m - 1.0) * 1e9

        w = torch.softmax(logits, dim=1)  # [I,3]
        return w.unsqueeze(-1)            # [I,3,1]


def apply_item_gate_concat(
    id_item: torch.Tensor,
    v_item: Optional[torch.Tensor],
    t_item: Optional[torch.Tensor],
    item_weight: torch.Tensor,   # [I,3,1]
) -> torch.Tensor:
    """
    Weighted concat for item representations
    """
    I = id_item.size(0)
    device = id_item.device
    w = item_weight.squeeze(-1)  # [I,3]

    reps: List[torch.Tensor] = []
    # id
    reps.append(id_item * w[:, 0:1].to(device=device, dtype=id_item.dtype))
    # v
    if v_item is not None:
        reps.append(v_item * w[:, 1:2].to(device=device, dtype=v_item.dtype))
    # t
    if t_item is not None:
        reps.append(t_item * w[:, 2:3].to(device=device, dtype=t_item.dtype))

    return torch.cat(reps, dim=1)

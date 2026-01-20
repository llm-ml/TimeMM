# time_user_gate.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def build_user_time_features_from_train_df(
    df_train: pd.DataFrame,
    uid_field: str,
    time_col: str,
    num_users: int,
    recent_window_sec: int = 86400,  # 1 day
) -> torch.FloatTensor:
    """
    Return user-level time features [num_users, F],ONLY uses training interactions.
    """
    # max_t per user (train)
    g = df_train.groupby(uid_field, as_index=False)[time_col]
    max_t = g.max().rename(columns={time_col: "max_t"})
    min_t = g.min().rename(columns={time_col: "min_t"})

    df = df_train[[uid_field, time_col]].merge(max_t, on=uid_field, how="left")
    df["age_sec"] = (df["max_t"] - df[time_col]).clip(lower=0).astype(np.float64)

    # per-user stats
    # mean / median age
    age_stats = df.groupby(uid_field)["age_sec"].agg(["mean", "median"]).reset_index()
    age_stats = age_stats.rename(columns={"mean": "mean_age", "median": "p50_age"})

    # span
    span = max_t.merge(min_t, on=uid_field, how="left")
    span["span_sec"] = (span["max_t"] - span["min_t"]).clip(lower=0).astype(np.float64)

    # recent fraction (how many interactions in last window)
    df["is_recent"] = (df[time_col] >= (df["max_t"] - recent_window_sec)).astype(np.float64)
    recent = df.groupby(uid_field)["is_recent"].mean().reset_index().rename(columns={"is_recent": "recent_frac"})

    # count
    cnt = df.groupby(uid_field).size().reset_index(name="n_inter").astype({uid_field: np.int64})

    # merge all
    feat = (
        age_stats.merge(span[[uid_field, "span_sec"]], on=uid_field, how="left")
        .merge(recent, on=uid_field, how="left")
        .merge(cnt, on=uid_field, how="left")
    )

    # init full table
    out = np.zeros((num_users, 4), dtype=np.float32)  # [mean_age, p50_age, span_sec, recent_frac]
    # map to rows
    u = feat[uid_field].to_numpy(np.int64)
    out[u, 0] = np.log1p(feat["mean_age"].to_numpy(np.float64)).astype(np.float32)
    out[u, 1] = np.log1p(feat["p50_age"].to_numpy(np.float64)).astype(np.float32)
    out[u, 2] = np.log1p(feat["span_sec"].to_numpy(np.float64)).astype(np.float32)
    out[u, 3] = feat["recent_frac"].to_numpy(np.float64).astype(np.float32)

    # simple normalize, optional but stabilizes
    eps = 1e-6
    mean = out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True) + eps
    newout = (out - mean) / std
    
    return torch.from_numpy(newout)#, out


class TimeAwareUserModalWeight(nn.Module):
    """
    Time-aware modulation for per-user modality weights.
    """
    def __init__(self, in_dim: int = 4, hidden: int = 32, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, user_time_feat: torch.Tensor) -> torch.Tensor:
        """
        user_time_feat: [U, F]
        """
        # add time-conditioned delta logits
        delta_logits = self.mlp(user_time_feat)          # [U,3]
        #delta_logits = delta_logits * self.alpha

        logits = delta_logits

        # normalize to probability simplex
        w = torch.softmax(logits, dim=1)                 # [U,3], sum=1, >=0
        
        # restore shape
        out_shape = "U31"
        if out_shape == "U31":
            return w.unsqueeze(-1)                       # [U,3,1]
        else:
            return w.unsqueeze(1)                        # [U,1,3]
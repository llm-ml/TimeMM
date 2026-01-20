# time_state_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd
import torch


@dataclass
class TimeStateCfg:
    """
    Build per-user and per-item temporal state vectors from TRAIN interactions only.
    """
    transform: Literal["raw", "log1p"] = "log1p"

    # tiny eps for safe division
    eps: float = 1e-12

    # Whether to include std features, slightly more compute, often useful
    include_std: bool = True


def _maybe_log1p(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return x.astype(np.float32)
    if mode == "log1p":
        return np.log1p(np.maximum(x, 0.0)).astype(np.float32)
    raise ValueError(f"Unknown transform={mode}")


def build_user_item_states_from_train_df(
    df_train: pd.DataFrame,
    uid_field: str,
    iid_field: str,
    time_col: str,
    num_user: int,
    num_item: int,
    device: torch.device,
    cfg: Optional[TimeStateCfg] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    most stable, evaluation-compatible temporal states
    any missing user item rows are zero vectors, this is safe fallback
    """
    cfg = cfg or TimeStateCfg()

    required = {uid_field, iid_field, time_col}
    missing = required - set(df_train.columns)
    if missing:
        raise ValueError(f"df_train missing columns: {sorted(missing)}")

    # keep only needed columns, ensure int64 timestamps
    df = df_train[[uid_field, iid_field, time_col]].copy()
    df[time_col] = df[time_col].astype(np.int64)
    
    # Sort by (u, t) to compute per-event gaps
    df_u = df.copy()
    df_u["_ord"] = np.arange(len(df_u), dtype=np.int64)
    df_u = df_u.sort_values([uid_field, time_col, "_ord"], kind="mergesort")

    # gap between consecutive events within user
    gap_u = df_u.groupby(uid_field)[time_col].diff().fillna(0).astype(np.int64)
    gap_u = gap_u.clip(lower=0)
    df_u["gap_u"] = gap_u.to_numpy(np.int64)

    # aggregates
    g_u = df_u.groupby(uid_field, as_index=True)

    u_cnt = g_u.size().astype(np.float64)  # n_inter
    u_min = g_u[time_col].min().astype(np.int64)
    u_max = g_u[time_col].max().astype(np.int64)
    u_span = (u_max - u_min).astype(np.float64)

    u_mean_gap = g_u["gap_u"].mean().astype(np.float64)

    # per-user age defined by user anchor
    # age_event = u_max - t_event
    # compute mean/std age via group transform
    u_max_map = u_max.rename("u_anchor")
    df_u = df_u.merge(u_max_map, left_on=uid_field, right_index=True, how="left")
    age_u_event = (df_u["u_anchor"].to_numpy(np.int64) - df_u[time_col].to_numpy(np.int64)).astype(np.int64)
    age_u_event = np.maximum(age_u_event, 0)
    df_u["age_u"] = age_u_event

    g_u2 = df_u.groupby(uid_field, as_index=True)
    u_mean_age = g_u2["age_u"].mean().astype(np.float64)

    # age_last_sec = u_anchor - last_t = 0 (by definition), still keep as feature for consistency
    u_age_last = np.zeros_like(u_cnt.to_numpy(), dtype=np.float64)

    if cfg.include_std:
        u_std_gap = g_u2["gap_u"].std(ddof=0).fillna(0.0).astype(np.float64)
        u_std_age = g_u2["age_u"].std(ddof=0).fillna(0.0).astype(np.float64)

    # pack user_state into [U, Fu]
    Fu = 7 if cfg.include_std else 5
    user_state = np.zeros((num_user, Fu), dtype=np.float32)

    # only fill users that exist
    u_idx = u_cnt.index.to_numpy(np.int64)
    # features (apply transform at the end for numerical stability)
    u_feat_list = [
        u_cnt.to_numpy(np.float64),
        u_span.to_numpy(np.float64),
        u_mean_gap.to_numpy(np.float64),
        u_age_last.astype(np.float64),
        u_mean_age.to_numpy(np.float64),
    ]
    if cfg.include_std:
        u_feat_list += [u_std_gap.to_numpy(np.float64), u_std_age.to_numpy(np.float64)]

    u_feat = np.stack(u_feat_list, axis=1)  # [n_users_present, Fu]
    u_feat = _maybe_log1p(u_feat, cfg.transform)
    user_state[u_idx] = u_feat

    # global anchor for item age
    global_anchor = int(df[time_col].max())

    df_i = df.copy()
    df_i["_ord"] = np.arange(len(df_i), dtype=np.int64)
    df_i = df_i.sort_values([iid_field, time_col, "_ord"], kind="mergesort")

    # gap between consecutive occurrences within item
    gap_i = df_i.groupby(iid_field)[time_col].diff().fillna(0).astype(np.int64)
    gap_i = gap_i.clip(lower=0)
    df_i["gap_i"] = gap_i.to_numpy(np.int64)

    g_i = df_i.groupby(iid_field, as_index=True)

    i_cnt = g_i.size().astype(np.float64)  # n_inter for item
    i_nuser = g_i[uid_field].nunique().astype(np.float64)

    i_min = g_i[time_col].min().astype(np.int64)
    i_max = g_i[time_col].max().astype(np.int64)
    i_span = (i_max - i_min).astype(np.float64)
    
    # global-age for item events like global_anchor - t_event
    age_i_last = (global_anchor - i_max).astype(np.float64)
    # mean age
    df_i["age_i_global"] = (global_anchor - df_i[time_col].to_numpy(np.int64)).astype(np.int64)
    df_i["age_i_global"] = df_i["age_i_global"].clip(lower=0)
    g_i2 = df_i.groupby(iid_field, as_index=True)
    i_mean_age = g_i2["age_i_global"].mean().astype(np.float64)

    if cfg.include_std:
        i_std_gap = g_i2["gap_i"].std(ddof=0).fillna(0.0).astype(np.float64)
        i_std_age = g_i2["age_i_global"].std(ddof=0).fillna(0.0).astype(np.float64)

    Fi = 7 if cfg.include_std else 5
    item_state = np.zeros((num_item, Fi), dtype=np.float32)

    i_idx = i_cnt.index.to_numpy(np.int64)
    i_feat_list = [
        i_cnt.to_numpy(np.float64),
        i_nuser.to_numpy(np.float64),
        i_span.to_numpy(np.float64),
        age_i_last.to_numpy(np.float64),
        i_mean_age.to_numpy(np.float64),
    ]
    if cfg.include_std:
        i_feat_list += [i_std_gap.to_numpy(np.float64), i_std_age.to_numpy(np.float64)]

    i_feat = np.stack(i_feat_list, axis=1)
    i_feat = _maybe_log1p(i_feat, cfg.transform)
    item_state[i_idx] = i_feat

    # to torch
    user_state_t = torch.tensor(user_state, dtype=torch.float32, device=device)
    item_state_t = torch.tensor(item_state, dtype=torch.float32, device=device)
    return user_state_t, item_state_t

# time_edge_weight_builder_multiscale.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch


@dataclass
class MultiScaleTimeWeightCfg:
    """
    Multi-scale time edge weighting
    edge_weight_k will be built for each tau in tau_age_list.
    """
    # which time vars, for now we can use age and keep flags for compatibility
    use_age: bool = True
    use_gap: bool = False

    # exp: f(t)=exp(-log1p(t)/tau)
    # inv: f(t)=1/(1+t/tau)
    transform: Literal["exp", "inv"] = "exp"
    
    # multi-scale params
    # Provide K taus explicitly. Example: [2.0, 6.0, 18.0]
    tau_age_list: Optional[List[float]] = None
    tau_age: float = 4.0
    
    # clip final weight
    clip_min: float = 1e-3
    clip_max: float = 1.0


def _factor(t_sec: np.ndarray, transform: str, tau: float) -> np.ndarray:
    """
    # Important, here we can also set multiple scales, and our online experiments showed that it also works very well
    # so in you practial applications, you can also change the scale of original-t in the scale of "Day", "Week", "Month"
    """
    t = np.maximum(t_sec.astype(np.float32), 0.0)   
    tau = float(max(tau, 1e-8))

    if transform == "exp":
        # exp(-log1p(t)/tau)  -- robust for seconds
        return np.exp(-np.log1p(t) / tau).astype(np.float32)
    elif transform == "inv":
        # 1/(1+t/tau)
        return (1.0 / (1.0 + t / tau)).astype(np.float32)
    else:
        return t / tau
        #raise ValueError(f"Unknown transform={transform}")


def build_edge_time_vars_from_train_df(
    df_train: pd.DataFrame,
    uid_field: str,
    iid_field: str,
    time_col: str,
) -> pd.DataFrame:
    """
    Return a per-(u,i) table
    """
    df = df_train[[uid_field, iid_field, time_col]].copy()
    df["_ord"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values([uid_field, time_col, "_ord"], kind="mergesort")

    # gap per EVENT within user
    gap_event = df.groupby(uid_field)[time_col].diff().fillna(0).astype(np.int64)
    gap_event = gap_event.clip(lower=0)
    df["gap_event_sec"] = gap_event

    # latest event per (u,i)
    latest = df.groupby([uid_field, iid_field], as_index=False).tail(1)

    # anchor per user (max train time)
    anchor = df.groupby(uid_field, as_index=True)[time_col].max().rename("t_anchor")
    latest = latest.merge(anchor, left_on=uid_field, right_index=True, how="left")

    t_ui = latest[time_col].to_numpy(np.int64)
    t_anchor = latest["t_anchor"].to_numpy(np.int64)
    age_sec = (t_anchor - t_ui).astype(np.int64)
    age_sec = np.maximum(age_sec, 0)

    out = pd.DataFrame({
        uid_field: latest[uid_field].to_numpy(),
        iid_field: latest[iid_field].to_numpy(),
        "t_ui": t_ui,
        "age_sec": age_sec,
        "gap_sec": latest["gap_event_sec"].to_numpy(np.int64),
    })
    return out


def build_edge_weight_for_ui_graph_multiscale(
    edge_index_1way_np: np.ndarray,     # [nnz, 2], item already offset by num_user
    num_user: int,
    device: torch.device,
    df_train: pd.DataFrame,
    uid_field: str,
    iid_field: str,
    time_col: str,
    cfg: Optional[MultiScaleTimeWeightCfg] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = cfg or MultiScaleTimeWeightCfg()

    edge_index_1way_np = np.asarray(edge_index_1way_np)
    assert edge_index_1way_np.ndim == 2 and edge_index_1way_np.shape[1] == 2, "edge_index_1way must be [nnz,2]"
    nnz = edge_index_1way_np.shape[0]

    # (u, i) for merge (restore item id)
    u_np = edge_index_1way_np[:, 0].astype(np.int64)
    i_np = (edge_index_1way_np[:, 1] - int(num_user)).astype(np.int64)

    # per-(u,i) time vars from train df
    ui_time = build_edge_time_vars_from_train_df(df_train, uid_field, iid_field, time_col)

    # merge to align with each edge
    edge_df = pd.DataFrame({uid_field: u_np, iid_field: i_np})
    edge_df = edge_df.merge(
        ui_time[[uid_field, iid_field, "age_sec", "gap_sec"]],
        on=[uid_field, iid_field],
        how="left",
    )

    # safety fallback: unseen edge -> extremely old
    edge_df["age_sec"] = edge_df["age_sec"].fillna(10**12).astype(np.int64)
    edge_df["gap_sec"] = edge_df["gap_sec"].fillna(0).astype(np.int64)

    age_sec = edge_df["age_sec"].to_numpy(np.int64)
    gap_sec = edge_df["gap_sec"].to_numpy(np.int64)

    # decide tau list
    if cfg.tau_age_list is None or len(cfg.tau_age_list) == 0:
        tau_list = [float(cfg.tau_age)]
    else:
        tau_list = [float(x) for x in cfg.tau_age_list]
    K = len(tau_list)

    # build K raw weights for 1-way edges
    if not cfg.use_age:
        w_1way_k = np.ones((K, nnz), dtype=np.float32)
    else:
        w_1way_k = np.stack(
            [_factor(age_sec, cfg.transform, tau) for tau in tau_list],
            axis=0
        ).astype(np.float32)

    w_1way_k = np.clip(w_1way_k, cfg.clip_min, cfg.clip_max).astype(np.float32)

    # symmetrize: [K, 2*nnz]
    w_1way_k_t = torch.tensor(w_1way_k, dtype=torch.float32, device=device)
    edge_weight_k = torch.cat([w_1way_k_t, w_1way_k_t], dim=1)

    age_t = torch.tensor(age_sec, dtype=torch.long, device=device)

    # keep a placeholder gap tensor for compatibility
    if cfg.use_gap:
        gap_t = torch.tensor(gap_sec, dtype=torch.long, device=device)
    else:
        gap_t = torch.zeros(nnz, dtype=torch.long, device=device)

    return edge_weight_k, age_t, gap_t

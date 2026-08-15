"""Regression metrics used in the manuscript evaluation."""

from __future__ import annotations
import numpy as np


def regression_metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if y.size == 0:
        raise ValueError("No finite paired observations.")
    e = p - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    bias = float(np.mean(e))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum((y - p) ** 2))
    r2 = float('nan') if sst == 0 else float(1.0 - sse / sst)
    pearson_r = float('nan')
    if y.size > 1 and np.std(y) > 0 and np.std(p) > 0:
        pearson_r = float(np.corrcoef(y, p)[0, 1])
    obs_sd = float(np.std(y, ddof=0))
    pred_sd = float(np.std(p, ddof=0))
    ratio = float('nan') if obs_sd == 0 else float(pred_sd / obs_sd)
    return {
        "n": int(y.size), "rmse": rmse, "mae": mae, "r2": r2,
        "pearson_r": pearson_r, "bias": bias,
        "obs_mean": float(np.mean(y)), "pred_mean": float(np.mean(p)),
        "obs_sd": obs_sd, "pred_sd": pred_sd, "pred_sd_ratio": ratio,
        "obs_min": float(np.min(y)), "obs_max": float(np.max(y)),
        "pred_min": float(np.min(p)), "pred_max": float(np.max(p)),
    }

"""Publication plotting functions for the sealed Davison evaluation."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .metrics import regression_metrics


def figure5_observed_vs_predicted(y, p0, p1, output_prefix):
    m0 = regression_metrics(y, p0)
    m1 = regression_metrics(y, p1)
    vals = np.concatenate([np.asarray(y), np.asarray(p0), np.asarray(p1)])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * 0.03 or 0.1
    lo, hi = lo - pad, hi + pad
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharex=True, sharey=True, constrained_layout=True)
    last = None
    for ax, pred, m, title in [
        (axes[0], p0, m0, "No acquisition-date dropout"),
        (axes[1], p1, m1, "Mild shared acquisition-date dropout"),
    ]:
        mask = np.isfinite(y) & np.isfinite(pred)
        last = ax.hexbin(np.asarray(y)[mask], np.asarray(pred)[mask], gridsize=75, bins="log", mincnt=1, cmap="viridis", linewidths=0)
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1.3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Observed within-AOI yield anomaly")
        ax.text(0.03, 0.97, f"RMSE = {m['rmse']:.3f}
MAE = {m['mae']:.3f}
r = {m['pearson_r']:.3f}
Bias = {m['bias']:.3f}
Pred./obs. SD = {m['pred_sd_ratio']:.3f}", transform=ax.transAxes, va="top", bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.92))
    axes[0].set_ylabel("Predicted within-AOI yield anomaly")
    cbar = fig.colorbar(last, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("Hexagonal-bin count (log scale)")
    fig.suptitle("Sealed Davison County External Test", fontweight="bold")
    prefix = Path(output_prefix)
    fig.savefig(prefix.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(prefix.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def figure6_distributions(y, p0, p1, output_prefix):
    m0 = regression_metrics(y, p0); m1 = regression_metrics(y, p1)
    vals = np.concatenate([np.asarray(y), np.asarray(p0), np.asarray(p1)])
    vals = vals[np.isfinite(vals)]
    bins = np.linspace(float(vals.min()), float(vals.max()), 121)
    fig, ax = plt.subplots(figsize=(8.2, 5.3), constrained_layout=True)
    ax.hist(y, bins=bins, density=True, histtype='step', linewidth=2, label=f"Observed (SD = {m0['obs_sd']:.3f})")
    ax.hist(p0, bins=bins, density=True, histtype='step', linewidth=2, label=f"No dropout (SD = {m0['pred_sd']:.3f}; ratio = {m0['pred_sd_ratio']:.3f})")
    ax.hist(p1, bins=bins, density=True, histtype='step', linewidth=2, label=f"Mild dropout (SD = {m1['pred_sd']:.3f}; ratio = {m1['pred_sd_ratio']:.3f})")
    ax.axvline(m0['obs_mean'], linestyle='--', linewidth=1)
    ax.set_xlabel("Within-AOI corn yield anomaly"); ax.set_ylabel("Probability density")
    ax.set_title("Sealed Davison County Anomaly Distributions", fontweight='bold')
    ax.legend(frameon=False)
    prefix = Path(output_prefix)
    fig.savefig(prefix.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(prefix.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

"""Preprocessing definitions explicitly supported by the publication evidence export."""

from __future__ import annotations
import numpy as np


def strict_clear_mask(valid_mask, in_bounds_mask, doy, udm2_clear):
    """Return the manuscript strict-clear boolean mask.

    Rule: valid_mask AND in_bounds_mask AND finite(DOY) AND
    finite(UDM2_CLEAR) AND UDM2_CLEAR == 1.0.
    """
    return (
        np.asarray(valid_mask, dtype=bool)
        & np.asarray(in_bounds_mask, dtype=bool)
        & np.isfinite(doy)
        & np.isfinite(udm2_clear)
        & (np.asarray(udm2_clear) == 1.0)
    )


def within_aoi_anomaly(yield_values, *, clip_training=False, clip=(-4.0, 4.0)):
    """Standardize yield within one AOI as (yield - AOI mean) / AOI SD.

    Population standard deviation (ddof=0) is used to match the sealed
    prediction-table diagnostics where observed anomaly SD is approximately 1.
    Training-only clipping can be requested explicitly.
    """
    y = np.asarray(yield_values, dtype=float)
    mean = np.nanmean(y)
    sd = np.nanstd(y, ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("AOI yield standard deviation must be positive and finite.")
    z = (y - mean) / sd
    if clip_training:
        z = np.clip(z, clip[0], clip[1])
    return z, mean, sd


def spatial_zscore(values, valid_mask=None, *, min_valid_cells=500, min_sd=1e-5):
    """Per-AOI, per-date feature standardization.

    Parameters
    ----------
    values : array-like, shape (n_cells, n_features)
        Feature values for one AOI on one acquisition date.
    valid_mask : array-like of bool, optional
        Cells eligible to define date-level spatial statistics.
    min_valid_cells : int
        Publication threshold: a date must have >=500 valid cells.
    min_sd : float
        Publication safeguard: SD values below 1e-5 are replaced by 1.0
        so the centered feature remains numerically stable rather than exploding.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if valid_mask is None:
        mask = np.all(np.isfinite(x), axis=1)
    else:
        mask = np.asarray(valid_mask, dtype=bool) & np.all(np.isfinite(x), axis=1)
    if int(mask.sum()) < int(min_valid_cells):
        raise ValueError(f"Acquisition date has only {mask.sum()} valid cells; need {min_valid_cells}.")
    mean = np.nanmean(x[mask], axis=0)
    sd = np.nanstd(x[mask], axis=0, ddof=0)
    safe_sd = np.where(np.isfinite(sd) & (sd >= min_sd), sd, 1.0)
    return (x - mean) / safe_sd, mean, safe_sd

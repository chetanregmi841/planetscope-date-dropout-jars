"""Reference implementation of the manuscript's shared acquisition-date dropout policy."""

from __future__ import annotations
import numpy as np


def shared_date_dropout(valid_dates, *, probability=0.20, min_dates=8, rng=None):
    """Select retained acquisition dates for training-time augmentation.

    This implements the locked policy-level description: acquisition dates are
    removed jointly (all features at a selected date), with dropout probability
    0.20, while retaining at least 8 valid dates. No dropout should be used at
    inference.

    Notes
    -----
    The archival export did not preserve the exact original random-sampling
    routine. This function is therefore a transparent *reference implementation*
    of the locked policy, not a claim of byte-identical original training code.
    """
    valid = np.asarray(valid_dates, dtype=bool).copy()
    idx = np.flatnonzero(valid)
    if idx.size <= min_dates:
        return valid
    rng = np.random.default_rng() if rng is None else rng
    keep = rng.random(idx.size) >= float(probability)
    if int(keep.sum()) < int(min_dates):
        # deterministically restore randomly selected dropped dates until min_dates
        dropped = np.flatnonzero(~keep)
        need = int(min_dates - keep.sum())
        restore = rng.choice(dropped, size=need, replace=False)
        keep[restore] = True
    out = np.zeros_like(valid)
    out[idx[keep]] = True
    return out

#!/usr/bin/env python
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planetscope_availability.metrics import regression_metrics

EXPECTED = {
    "no_dropout": {"n": 140856, "rmse": 1.292631148, "mae": 1.128608549, "r2": -0.670895277, "pearson_r": 0.259001957, "bias": -0.848408493, "pred_sd_ratio": 0.124167526},
    "mild_dropout": {"n": 140856, "rmse": 1.002273, "mae": 0.773469, "r2": -0.004550, "pearson_r": 0.254447, "bias": -0.229090, "pred_sd_ratio": 0.124788},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-dropout', required=True)
    ap.add_argument('--mild-dropout', required=True)
    ap.add_argument('--json-out')
    args = ap.parse_args()
    a = pd.read_parquet(args.no_dropout)
    b = pd.read_parquet(args.mild_dropout)
    required = {'OBSERVED_ANOMALY', 'PREDICTED_ANOMALY'}
    if not required.issubset(a.columns) or not required.issubset(b.columns):
        raise SystemExit(f"Prediction tables must contain {sorted(required)}")
    if len(a) != len(b):
        raise SystemExit('FAIL: row counts differ')
    y0 = a['OBSERVED_ANOMALY'].to_numpy(float)
    y1 = b['OBSERVED_ANOMALY'].to_numpy(float)
    maxdiff = float(np.nanmax(np.abs(y0 - y1)))
    print(f"paired rows: {len(a):,}")
    print(f"max observed-target difference: {maxdiff:.12f}")
    m0 = regression_metrics(y0, a['PREDICTED_ANOMALY'])
    m1 = regression_metrics(y1, b['PREDICTED_ANOMALY'])
    for name, m in [('no_dropout', m0), ('mild_dropout', m1)]:
        print(f"\n{name}")
        for k in ['n','rmse','mae','r2','pearson_r','bias','pred_sd_ratio']:
            print(f"  {k}: {m[k]}")
    if maxdiff != 0.0:
        raise SystemExit('FAIL: observed targets are not identical by row order')
    result = {'max_observed_target_difference': maxdiff, 'no_dropout': m0, 'mild_dropout': m1}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()

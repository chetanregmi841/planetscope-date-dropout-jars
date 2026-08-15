#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planetscope_availability.plotting import figure5_observed_vs_predicted, figure6_distributions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-dropout', required=True)
    ap.add_argument('--mild-dropout', required=True)
    ap.add_argument('--output-dir', default='outputs')
    args = ap.parse_args()
    a = pd.read_parquet(args.no_dropout)
    b = pd.read_parquet(args.mild_dropout)
    y0 = a['OBSERVED_ANOMALY'].to_numpy(float)
    y1 = b['OBSERVED_ANOMALY'].to_numpy(float)
    if len(a) != len(b) or float(np.nanmax(np.abs(y0-y1))) != 0.0:
        raise SystemExit('Prediction tables are not row-paired with identical observed targets.')
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    p0 = a['PREDICTED_ANOMALY'].to_numpy(float)
    p1 = b['PREDICTED_ANOMALY'].to_numpy(float)
    figure5_observed_vs_predicted(y0, p0, p1, out/'Figure_5_Davison_Observed_vs_Predicted_FINAL')
    figure6_distributions(y0, p0, p1, out/'Figure_6_Davison_Anomaly_Distributions_FINAL')
    print(f'Wrote figures to {out.resolve()}')

if __name__ == '__main__':
    main()

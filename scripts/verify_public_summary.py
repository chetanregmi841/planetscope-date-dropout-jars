#!/usr/bin/env python
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
for p in sorted((root/'data'/'public_summary').glob('*.csv')):
    df = pd.read_csv(p)
    print(f"{p.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.to_string(index=False))
    print()

# PlanetScope availability-aware corn yield-anomaly modeling

Publication companion repository for:

> **Shared acquisition-date dropout improves external PlanetScope corn yield-anomaly prediction under variable image availability**  
> Chetan Regmi — *Journal of Applied Remote Sensing* submission.

## What this repository contains

This is a **sanitized reproducibility companion** for the manuscript. It contains:

- locked model architecture metadata recovered from the selected checkpoints;
- strict-clear preprocessing and within-AOI anomaly definitions;
- the mild shared acquisition-date dropout policy used in the manuscript;
- independent metric recomputation and paired-table integrity checks;
- publication-figure generation scripts for the sealed Davison evaluation;
- nonrestricted aggregate metric tables used in the manuscript;
- environment and data-schema documentation.

It intentionally does **not** include restricted row-level yield data, PlanetScope imagery or derived row-level model inputs, credentials, local drive paths, or proprietary data-provider material.

## Study design

County roles were fixed before the sealed external evaluation:

- **Aurora County:** dense training subset plus a disjoint moderate holdout used for policy selection;
- **Sanborn County:** development/validation comparison;
- **Davison County:** sealed external test, untouched until final evaluation.

The manuscript compares:

1. **No acquisition-date dropout**; and
2. **Mild shared acquisition-date dropout**, applied during training only with probability `0.20`, while retaining at least `8` valid dates.

Inference always uses naturally observed acquisition availability.

## Locked model configuration

- Features per date: `14`
- Maximum temporal slots: `82`
- Transformer embedding dimension: `64`
- Attention heads: `4`
- Encoder layers: `2`
- Feed-forward dimension: `128`
- Internal model dropout: `0.10`
- Parameter count represented in the selected state dict: `74,497`
- Target: within-AOI yield anomaly `(yield - AOI mean) / AOI SD`
- Training-only target clipping: `[-4, 4]`
- Minimum clear observations per modeled row: `5`
- Minimum valid cells per AOI/date for spatial normalization: `500`
- Minimum feature SD safeguard: `1e-5`

The 14 features are:

`B01_COASTAL, B02_BLUE, B03_GREEN_I, B04_GREEN, B05_YELLOW, B06_RED, B07_REDEDGE, B08_NIR, NDVI, GNDVI, NDRE, EVI, SAVI, NDWI_GREEN_NIR`.

## Locked sealed Davison results

| Condition | n | RMSE | MAE | R² | Pearson r | Bias | Pred./obs. SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| No dropout | 140,856 | 1.292631 | 1.128609 | -0.670895 | 0.259002 | -0.848408 | 0.124168 |
| Mild shared date dropout | 140,856 | 1.002273 | 0.773469 | -0.004550 | 0.254447 | -0.229090 | 0.124788 |

All eight Davison field-year AOIs had lower RMSE under mild dropout.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Recompute the sealed-test metrics when authorized row-level prediction tables are available locally:

```bash
python scripts/recompute_davison_metrics.py \
  --no-dropout /path/to/BEST_DAVISON_SPATIAL_ANOMALY_PREDICTIONS.parquet \
  --mild-dropout /path/to/BEST_DAVISON_MILD_DROPOUT_PREDICTIONS.parquet
```

Generate manuscript Figures 5 and 6:

```bash
python scripts/generate_davison_figures.py \
  --no-dropout /path/to/BEST_DAVISON_SPATIAL_ANOMALY_PREDICTIONS.parquet \
  --mild-dropout /path/to/BEST_DAVISON_MILD_DROPOUT_PREDICTIONS.parquet \
  --output-dir outputs
```

## Important reproducibility note

The publication evidence export preserved the selected checkpoint architecture and state-dict shapes, the date-dropout probability, target definition, preprocessing thresholds, and evaluation outputs. It did **not** preserve the exact original optimizer, learning rate, batch size, loss function, activation choices, or sequence-pooling implementation from the final training script. Therefore:

- the evaluation and figure scripts in this repository reproduce the reported metrics from authorized prediction tables;
- `reference_model.py` reproduces the **parameterized architecture and parameter count** encoded by the selected checkpoint metadata, but its forward-path activation/pooling choices are explicitly marked as a reference implementation rather than an archival copy of the original training source;
- exact end-to-end retraining should not be claimed until the original training script is recovered and compared.

This limitation is deliberate: unknown details are not fabricated.

## Data availability

The row-level yield data and PlanetScope-derived model inputs are subject to research data-access restrictions and third-party licensing/data-provider agreements and are not redistributed here. See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

## Repository license

No software license has been selected in this package. Add an appropriate license only after confirming institutional/project permissions for public code release.

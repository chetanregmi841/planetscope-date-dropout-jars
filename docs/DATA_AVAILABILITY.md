# Data availability

The row-level yield data and PlanetScope-derived model inputs used in this study are subject to research data-access restrictions and third-party licensing or data-provider agreements. They are **not redistributed in this repository**.

Where possible, access requires approval from the applicable data providers and institution. Readers may contact the corresponding author for information on the access pathway.

This repository contains only nonrestricted aggregate metrics and code that can operate on authorized local copies of the prediction tables.

## Expected local prediction-table schema

The sealed Davison figure/metric scripts expect two Parquet files with identical row order and at minimum these columns:

- `OBSERVED_ANOMALY`
- `PREDICTED_ANOMALY`

The locked prediction exports also contained:

- `SUBSET`
- `AOI_KEY`
- `YEAR`
- `CNTY`
- `FLD`
- `MODEL_ROW_INDEX`
- `OBSERVED_YIELD_BU_AC`
- `AOI_YIELD_MEAN_BU_AC`
- `AOI_YIELD_SD_BU_AC`
- `N_NORMALIZED_OBS`
- `N_USABLE_DATES`
- `N_REJECTED_DATES`

Do not commit the restricted row-level files to a public repository.

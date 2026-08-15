"""Locked publication configuration recovered from checkpoint metadata."""

FEATURES = [
    "B01_COASTAL", "B02_BLUE", "B03_GREEN_I", "B04_GREEN",
    "B05_YELLOW", "B06_RED", "B07_REDEDGE", "B08_NIR",
    "NDVI", "GNDVI", "NDRE", "EVI", "SAVI", "NDWI_GREEN_NIR",
]

LOCKED_CONFIG = {
    "crop": "Corn",
    "n_features": 14,
    "max_slots": 82,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "dim_feedforward": 128,
    "model_dropout": 0.10,
    "date_dropout_probability": 0.20,
    "minimum_valid_dates_after_dropout": 8,
    "minimum_clear_observations_per_row": 5,
    "minimum_valid_cells_per_date": 500,
    "minimum_feature_sd": 1e-5,
    "training_target_clip": (-4.0, 4.0),
    "target_definition": "Within-AOI yield anomaly: (yield - AOI mean) / AOI SD",
    "strict_clear_rule": (
        "valid_mask AND in_bounds_mask AND finite(DOY) AND "
        "finite(UDM2_CLEAR) AND UDM2_CLEAR == 1.0"
    ),
    "development_split": (
        "Aurora dense train; Sanborn validation; Aurora moderate holdout; Davison untouched"
    ),
    "selected_checkpoint_epochs": {"no_dropout": 7, "mild_dropout": 24},
    "expected_parameter_count": 74497,
}

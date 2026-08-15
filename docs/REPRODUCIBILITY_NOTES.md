# Reproducibility notes

## Authoritative locked elements

The publication evidence export supports the following without inference:

- model state-dict parameter count: 74,497;
- 14 features and their names;
- transformer dimensions: 64 embedding, 4 heads, 2 encoder layers, 128 feed-forward dimension, 0.10 internal dropout;
- mild shared date-dropout probability: 0.20;
- at least 8 dates retained after training-time dropout;
- no artificial date dropout at inference;
- strict-clear rule and minimum 5 clear observations per row;
- minimum 500 valid cells per AOI/date for spatial normalization;
- minimum feature SD safeguard 1e-5;
- within-AOI target definition and training-only clipping [-4, 4];
- sealed-test checkpoint epochs 7 and 24;
- all reported pooled and AOI-level evaluation metrics.

## Not recovered from the publication export

The exact original final-training implementation did not expose the optimizer, learning rate, batch size, loss function, random seed(s), activation functions, or the exact sequence-pooling method. These are not invented here.

`reference_model.py` therefore reconstructs the parameterized layer shapes and exact parameter count but labels activation/pooling choices as reference choices. Use the metric and plotting scripts for direct verification of reported results from authorized prediction tables.

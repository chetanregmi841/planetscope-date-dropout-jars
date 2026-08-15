import numpy as np
from planetscope_availability.metrics import regression_metrics

def test_perfect_prediction():
    y = np.array([-1.0, 0.0, 1.0])
    m = regression_metrics(y, y)
    assert m['rmse'] == 0.0
    assert m['mae'] == 0.0
    assert m['bias'] == 0.0
    assert abs(m['r2'] - 1.0) < 1e-12
    assert abs(m['pearson_r'] - 1.0) < 1e-12
    assert abs(m['pred_sd_ratio'] - 1.0) < 1e-12

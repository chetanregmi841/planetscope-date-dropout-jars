import numpy as np
from planetscope_availability.preprocessing import strict_clear_mask, within_aoi_anomaly

def test_strict_clear():
    out = strict_clear_mask([1,1,1], [1,1,0], [100, np.nan, 120], [1.0,1.0,1.0])
    assert out.tolist() == [True, False, False]

def test_anomaly_sd_one():
    z, _, _ = within_aoi_anomaly([1.0, 2.0, 3.0])
    assert abs(float(np.std(z, ddof=0)) - 1.0) < 1e-12

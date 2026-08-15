import pytest

def test_reference_parameter_count():
    torch = pytest.importorskip('torch')
    from planetscope_availability.reference_model import AvailabilityAwareTransformer, parameter_count
    assert parameter_count(AvailabilityAwareTransformer()) == 74497

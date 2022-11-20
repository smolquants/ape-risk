import numpy as np
from hypothesis import given

from ape_risk import strategies


@given(strategies.sims(dist_type="norm", num_points=100000, params=[0, 1]))
def test_sims(s):
    assert s.shape == (100000, 1)
    assert isinstance(s, np.ndarray)

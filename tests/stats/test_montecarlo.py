import numpy as np
import pytest
from scipy import stats  # type: ignore


def test_dist(mc):
    # test succeeds when dist type supported by stats
    assert mc.dist == getattr(stats, mc.dist_type)

    # test fails when not supported by stats
    mc.dist_type = "fake_dist"
    with pytest.raises(Exception):
        _ = mc.dist


def test_sims(mc):
    # generate samples and check return shape is correct
    nd_samples = mc.sims()
    assert isinstance(nd_samples, np.ndarray)
    assert nd_samples.shape == (mc.num_points, mc.num_sims)

    # check loc, scale of all samples (assumes iid)
    # flatten nd array
    size = mc.num_points * mc.num_sims
    nd_samples_reshaped = np.reshape(nd_samples, size)

    # fit and check params close
    params = mc.dist.fit(nd_samples_reshaped)
    np.testing.assert_allclose(params, mc.params, rtol=1e-2)

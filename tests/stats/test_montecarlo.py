import numpy as np
import pytest

from ape_risk.stats.montecarlo import MonteCarlo


@pytest.fixture
def mc():
    return MonteCarlo()


def test_generate_rvs(mc, norm):
    num_points = 100000
    num_sims = 10
    args = [0.1, 0.001]  # loc, scale

    # generate samples and check return shape is correct
    nd_samples = mc.generate_rvs(*args, dist=norm, num_points=num_points, num_sims=num_sims)
    assert isinstance(nd_samples, np.ndarray)
    assert nd_samples.shape == (num_points, num_sims)

    # check loc, scale of all samples (assumes iid)
    # flatten nd array
    size = num_points * num_sims
    nd_samples_reshaped = np.reshape(nd_samples, size)

    # fit and check params close
    params = norm.fit(nd_samples_reshaped)
    np.testing.assert_allclose(params, args, rtol=1e-2)

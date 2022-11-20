import numpy as np
from hypothesis import given
from scipy import stats  # type: ignore

from ape_risk import strategies
from ape_risk.stats import MonteCarlo


def hist_data():
    dist_type = "norm"
    num_points = 200000
    params = [0.001, 0.005]  # loc, scale

    mc = MonteCarlo(dist_type=dist_type, num_points=num_points, num_sims=1)
    mc.freeze(params)
    log_p = mc.sims()

    p0 = 1.0
    return (p0 * np.exp(np.cumsum(log_p))).tolist()


@given(strategies.sims(dist_type="norm", num_points=100000, params=[0, 1]))
def test_sims_fuzz(s):
    assert s.shape == (100000, 1)
    assert isinstance(s, np.ndarray)


@given(strategies.gbms(initial_value=1.0, num_points=100000, params=[0.001, 0.005]))
def test_gbms_param_fuzz(p):
    assert p.shape == (100000, 1)
    assert isinstance(p, np.ndarray)

    # check distr of p is close to log normal with params
    dlog_p = np.diff(np.log(p.T))
    fit_params = stats.norm.fit(dlog_p)
    np.testing.assert_allclose(fit_params, [0.001, 0.005], rtol=2e-1)  # mu tol is not great


@given(strategies.gbms(initial_value=1.0, num_points=100000, params=[0, 1], hist_data=hist_data()))
def test_gbms_hist_fuzz(p):
    assert p.shape == (100000, 1)
    assert isinstance(p, np.ndarray)

    # check distr of p is close to log normal with params
    dlog_p = np.diff(np.log(p.T))
    fit_params = stats.norm.fit(dlog_p)
    np.testing.assert_allclose(fit_params, [0.001, 0.005], rtol=2e-1)  # mu tol is not great

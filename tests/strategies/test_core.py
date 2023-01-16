import numpy as np
from hypothesis import given
from scipy import stats  # type: ignore

from ape_risk import strategies
from ape_risk.stats import MonteCarlo, MultivariateMonteCarlo


def hist_data():
    dist_type = "norm"
    num_points = 200000
    params = [0.001, 0.005]  # loc, scale

    mc = MonteCarlo(dist_type=dist_type, num_points=num_points, num_sims=1)
    mc.freeze(params)
    log_p = mc.sims()

    p0 = 1.0
    return (p0 * np.exp(np.cumsum(log_p))).tolist()


def scale():
    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    return np.linalg.cholesky(C).tolist()


def multi_hist_data():
    dist_type = "norm"
    num_points = 200000
    num_rvs = 3
    params = [0, 0.005]  # loc, scale

    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    scale = np.linalg.cholesky(C)
    shift = np.asarray([0.001, 0.002, 0.003])

    mmc = MultivariateMonteCarlo(
        dist_type=dist_type, num_points=num_points, num_rvs=num_rvs, num_sims=1
    )
    mmc.freeze(params)
    mmc.mix(scale, shift)
    log_p = mmc.sims().reshape(num_points, num_rvs)

    p0 = 1.0
    return (p0 * np.exp(np.cumsum(log_p, axis=0))).tolist()


@given(strategies.sims(dist_type="norm", num_points=100000, params=[0, 1]))
def test_sims_fuzz(s):
    assert s.shape == (100000, 1)
    assert isinstance(s, np.ndarray)


@given(
    strategies.multi_sims(
        dist_type="norm",
        num_points=100000,
        num_rvs=3,
        params=[0, 1],
        scale=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        shift=[0, 0, 0],
    )
)
def test_multi_sims_fuzz(s):
    assert s.shape == (100000, 1, 3)
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


@given(
    strategies.multi_gbms(
        initial_value=1.0,
        num_points=100000,
        num_rvs=3,
        params=[0, 0.005],
        scale=scale(),
        shift=[0.001, 0.002, 0.003],
    )
)
def test_multi_gbms_param_fuzz(p):
    assert p.shape == (100000, 1, 3)
    assert isinstance(p, np.ndarray)

    # check distr of p is close to log normal with params
    dlog_p = np.diff(np.log(p.reshape(100000, 3)), axis=0)
    fit_shift = np.mean(dlog_p, axis=0)
    C = np.cov(dlog_p.T)
    fit_scale = np.linalg.cholesky(C)

    np.testing.assert_allclose(
        fit_scale, np.asarray(scale()) * 0.005, rtol=2e-1
    )  # cov tol is not great
    np.testing.assert_allclose(fit_shift, [0.001, 0.002, 0.003], 2e-1)  # mu tol is not great


@given(
    strategies.multi_gbms(
        initial_value=1.0,
        num_points=100000,
        num_rvs=3,
        params=[0, 1],
        scale=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        shift=[0, 0, 0],
        hist_data=multi_hist_data(),
    )
)
def test_multi_gbms_hist_fuzz(p):
    assert p.shape == (100000, 1, 3)
    assert isinstance(p, np.ndarray)

    # check distr of p is close to log normal with params
    dlog_p = np.diff(np.log(p.reshape(100000, 3)), axis=0)
    fit_shift = np.mean(dlog_p, axis=0)
    C = np.cov(dlog_p.T)
    fit_scale = np.linalg.cholesky(C)

    np.testing.assert_allclose(
        fit_scale, np.asarray(scale()) * 0.005, rtol=2e-1
    )  # cov tol is not great
    np.testing.assert_allclose(fit_shift, [0.001, 0.002, 0.003], 2e-1)  # mu tol is not great

import numpy as np
import pytest
from hypothesis.internal.conjecture.data import ConjectureData

from ape_risk.stats import MonteCarlo, MultivariateMonteCarlo
from ape_risk.strategies import MultivariateSimulationStrategy, SimulationStrategy


def test_init_without_hist_data():
    dist_type = "norm"
    num_points = 100000
    params = [0.1, 0.001]  # loc, scale
    sim_strat = SimulationStrategy(
        dist_type=dist_type,
        num_points=num_points,
        params=params,
    )

    # check internal mc matches params to sim strat
    assert sim_strat._mc.dist_type == dist_type
    assert sim_strat._mc.num_points == num_points
    assert sim_strat._mc.num_sims == 1
    np.testing.assert_equal(sim_strat._mc.params, params)


def test_init_with_hist_data():
    dist_type = "norm"
    num_points = 100000
    params = [0.1, 0.001]  # loc, scale

    mc = MonteCarlo(dist_type=dist_type, num_points=num_points, num_sims=1)
    mc.freeze(params)
    hist_data = mc.sims().tolist()

    sim_strat = SimulationStrategy(
        dist_type=dist_type,
        num_points=num_points,
        params=[0, 1],  # initial loc, scale
        hist_data=hist_data,
    )

    # check internal mc matches params to sim strat
    assert sim_strat._mc.dist_type == dist_type
    assert sim_strat._mc.num_points == num_points
    assert sim_strat._mc.num_sims == 1

    # check params fit from historical are same as norm rv fit
    np.testing.assert_allclose(sim_strat._mc.params, params, rtol=1e-2)


def test_init_with_hist_data_raises_when_num_points_gt_len_hist():
    dist_type = "norm"
    num_points = 100000
    params = [0.1, 0.001]  # loc, scale

    mc = MonteCarlo(dist_type=dist_type, num_points=num_points, num_sims=1)
    mc.freeze(params)
    hist_data = mc.sims().tolist()

    with pytest.raises(ValueError):
        _ = SimulationStrategy(
            dist_type=dist_type,
            num_points=num_points + 1,
            params=[0, 1],  # initial loc, scale
            hist_data=hist_data,
        )


def test_do_draw():
    dist_type = "norm"
    num_points = 100000
    params = [0.1, 0.001]  # loc, scale
    sim_strat = SimulationStrategy(
        dist_type=dist_type,
        num_points=num_points,
        params=params,
    )

    # SEE:https://github.com/HypothesisWorks/hypothesis/blob/master/hypothesis-python/tests/conjecture/test_utils.py  # noqa: E501
    data = ConjectureData.for_buffer(bytes(8))
    draw = sim_strat.do_draw(data=data)
    assert draw.shape == (100000, 1)
    assert isinstance(draw, np.ndarray)

    # check draw has fit close to params
    sim_strat._mc.fit(draw)
    np.testing.assert_allclose(sim_strat._mc.params, params, rtol=1e-2)


def test_multi_init_without_hist_data():
    dist_type = "norm"
    num_points = 100000
    num_rvs = 3
    params = [0, 0.001]

    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    scale = np.linalg.cholesky(C).tolist()
    shift = [0.1, 0.2, 0.3]
    sim_strat = MultivariateSimulationStrategy(
        dist_type=dist_type,
        num_points=num_points,
        num_rvs=num_rvs,
        params=params,
        scale=scale,
        shift=shift,
    )

    # check internal mmc matches params, scale, shift to sim strat
    assert sim_strat._mmc.dist_type == dist_type
    assert sim_strat._mmc.num_points == num_points
    assert sim_strat._mmc.num_sims == 1
    assert sim_strat._mmc.num_rvs == num_rvs
    np.testing.assert_equal(sim_strat._mmc.params, params)
    np.testing.assert_equal(sim_strat._mmc.scale, scale)
    np.testing.assert_equal(sim_strat._mmc.shift, shift)


def test_multi_init_with_hist_data():
    dist_type = "norm"
    num_points = 100000
    num_rvs = 3
    params = [0, 0.001]

    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    scale = np.linalg.cholesky(C)
    shift = np.asarray([0.1, 0.2, 0.3])

    mmc = MultivariateMonteCarlo(
        dist_type=dist_type,
        num_points=num_points,
        num_sims=1,
        num_rvs=num_rvs,
    )
    mmc.freeze(params)
    mmc.mix(scale, shift)
    hist_data = mmc.sims().reshape(num_points, num_rvs).tolist()

    sim_strat = MultivariateSimulationStrategy(
        dist_type=dist_type,
        num_points=num_points,
        num_rvs=num_rvs,
        params=[0, 1],
        scale=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        shift=[0, 0, 0],
        hist_data=hist_data,
    )

    # check internal mmc matches params to sim strat
    np.testing.assert_equal(sim_strat._mmc.params, [0, 1])
    np.testing.assert_allclose(sim_strat._mmc.scale, scale * params[1], rtol=5e-2)
    np.testing.assert_allclose(sim_strat._mmc.shift, shift, rtol=5e-2)


def test_multi_init_with_hist_data_raises_when_num_points_gt_len_hist():
    dist_type = "norm"
    num_points = 100000
    num_rvs = 3
    params = [0, 0.001]  # loc, scale

    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    scale = np.linalg.cholesky(C)
    shift = np.asarray([0.1, 0.2, 0.3])

    mmc = MultivariateMonteCarlo(
        dist_type=dist_type,
        num_points=num_points,
        num_sims=1,
        num_rvs=num_rvs,
    )
    mmc.freeze(params)
    mmc.mix(scale, shift)
    hist_data = mmc.sims().reshape(num_points, num_rvs).tolist()

    with pytest.raises(ValueError):
        _ = MultivariateSimulationStrategy(
            dist_type=dist_type,
            num_points=num_points + 1,
            num_rvs=num_rvs,
            params=[0, 1],  # initial loc, scale
            scale=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # initial scale
            shift=[0, 0, 0],  # initial shift
            hist_data=hist_data,
        )


def test_multi_do_draw():
    dist_type = "norm"
    num_points = 100000
    num_rvs = 3
    params = [0, 0.001]

    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    scale = np.linalg.cholesky(C).tolist()
    shift = [0.1, 0.2, 0.3]
    sim_strat = MultivariateSimulationStrategy(
        dist_type=dist_type,
        num_points=num_points,
        num_rvs=num_rvs,
        params=params,
        scale=scale,
        shift=shift,
    )

    # SEE:https://github.com/HypothesisWorks/hypothesis/blob/master/hypothesis-python/tests/conjecture/test_utils.py  # noqa: E501
    data = ConjectureData.for_buffer(bytes(8))
    draw = sim_strat.do_draw(data=data)
    assert draw.shape == (100000, 1, 3)
    assert isinstance(draw, np.ndarray)

    # check draw has fit close to params
    sim_strat._mmc.fit(draw.reshape(100000, 3))
    np.testing.assert_equal(sim_strat._mmc.params, [0, 1])
    np.testing.assert_allclose(sim_strat._mmc.scale, np.asarray(scale) * params[1], rtol=5e-2)
    np.testing.assert_allclose(sim_strat._mmc.shift, np.asarray(shift), rtol=5e-2)

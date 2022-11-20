import numpy as np
import pytest

from ape_risk.stats import MonteCarlo
from ape_risk.strategies import SimulationStrategy


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


# TODO: def test_do_draw():

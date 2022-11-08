import pytest

from ape_risk.stats.montecarlo import MonteCarlo


@pytest.fixture
def mc():
    dist_type = "norm"
    params = [0.1, 0.001]  # loc, scale
    num_points = 100000
    num_sims = 10
    return MonteCarlo(dist_type=dist_type, params=params, num_points=num_points, num_sims=num_sims)

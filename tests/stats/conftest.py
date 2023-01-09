import pytest

from ape_risk.stats.montecarlo import MonteCarlo, MultivariateMonteCarlo


@pytest.fixture
def mc():
    dist_type = "norm"
    num_points = 100000
    num_sims = 10
    return MonteCarlo(dist_type=dist_type, num_points=num_points, num_sims=num_sims)


@pytest.fixture
def mmc():
    dist_type = "norm"
    num_points = 100000
    num_sims = 10
    num_rvs = 3
    return MultivariateMonteCarlo(
        dist_type=dist_type,
        num_points=num_points,
        num_sims=num_sims,
        num_rvs=num_rvs,
    )

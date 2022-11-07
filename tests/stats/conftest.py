import pytest
from scipy import stats  # type: ignore


@pytest.fixture(scope="module")
def norm():
    return stats.norm

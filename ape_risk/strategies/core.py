from typing import List, Optional

import numpy.typing as npt
from hypothesis import strategies as st
from hypothesis.internal.validation import check_type
from hypothesis.strategies._internal.utils import cacheable, defines_strategy


@cacheable
@defines_strategy()
def sims(
    *,
    dist_type: str,
    num_points: int,
    params: List,
    hist_data: Optional[List] = None,
) -> st.SearchStrategy[npt.ArrayLike]:
    """
    Generates instances of ``np.ndarray``. The generated random instances
    are individual runs of a Monte Carlo sim driven by the
    specified distribution.

    Args:
        dist_type (str): The continuous distribution to sample from.
        num_points (int): The number of points to generate for each sim.
        params (List): The parameter arguments (e.g. loc, scale) of the random variable.
        hist_data (Optional[List]): Historical data to fit the random variable params from.

    Returns:
        :class:`hypothesis.strategies.SearchStrategy[numpy.typing.ArrayLike]`
    """
    check_type(str, dist_type, "dist_type")
    check_type(int, num_points, "num_points")
    check_type(list, params, "params")
    if hist_data is not None:
        check_type(list, hist_data, "hist_data")

    from ape_risk.strategies.simulation import SimulationStrategy

    return SimulationStrategy(
        dist_type=dist_type, num_points=num_points, params=params, hist_data=hist_data
    )
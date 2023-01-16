from typing import List, Optional

import numpy as np
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


@cacheable
@defines_strategy()
def multi_sims(
    *,
    dist_type: str,
    num_points: int,
    num_rvs: int,
    params: List,
    scale: List[List],
    shift: List,
    hist_data: Optional[List[List]] = None,
) -> st.SearchStrategy[npt.ArrayLike]:
    """
    Generates instances of ``np.ndarray``. The generated random instances
    are individual runs of a multivariate Monte Carlo sim driven by the
    specified distribution, with covariance determined by the given
    scale matrix (Cholesky decomposition of covariance matrix).

    Args:
        dist_type (str): The continuous distribution to sample from.
        num_points (int): The number of points to generate for each sim.
        params (List): The base parameter arguments (e.g. loc, scale) of the random variables.
        scale (List[List]): The scale matrix to mix random variables via affine transformation.
        shift (List): The shift vector to translate random variables via affine transformation.
        hist_data (Optional[List[List]]): Historical data to fit the random variable params from.

    Returns:
        :class:`hypothesis.strategies.SearchStrategy[numpy.typing.ArrayLike]`
    """
    check_type(str, dist_type, "dist_type")
    check_type(int, num_points, "num_points")
    check_type(int, num_rvs, "num_rvs")
    check_type(list, params, "params")
    check_type(list, scale, "scale")
    check_type(list, shift, "shift")
    if hist_data is not None:
        check_type(list, hist_data, "hist_data")

    from ape_risk.strategies.simulation import MultivariateSimulationStrategy

    return MultivariateSimulationStrategy(
        dist_type=dist_type,
        num_points=num_points,
        num_rvs=num_rvs,
        params=params,
        scale=scale,
        shift=shift,
        hist_data=hist_data,
    )


@cacheable
@defines_strategy()
def gbms(
    *,
    initial_value: float,
    num_points: int,
    params: List,
    hist_data: Optional[List] = None,
) -> st.SearchStrategy[npt.ArrayLike]:
    """
    Generates instances of ``np.ndarray``. The generated random instances
    are individual runs of a Monte Carlo sim driven by Geometric Brownian motion.

    Args:
        initial_value (float): The initial value of each sim.
        num_points (int): The number of points to generate for each sim.
        params (List): The parameter arguments (e.g. loc, scale) of the random variable.
        hist_data (Optional[List]): Historical data to fit the random variable params from.

    Returns:
        :class:`hypothesis.strategies.SearchStrategy[numpy.typing.ArrayLike]`
    """
    check_type(float, initial_value, "initial_value")
    check_type(int, num_points, "num_points")
    check_type(list, params, "params")
    if hist_data is not None:
        check_type(list, hist_data, "hist_data")

        # fit to log differences: log(p[i+1] / p[i])
        hist_data = np.diff(np.log(np.asarray(hist_data))).tolist()

    from ape_risk.strategies.simulation import SimulationStrategy

    strat = SimulationStrategy(
        dist_type="norm", num_points=num_points, params=params, hist_data=hist_data
    )

    def pack(x: npt.ArrayLike) -> npt.ArrayLike:
        return initial_value * np.exp(np.cumsum(x, axis=0))  # axis=0 sums over rows

    return strat.map(pack)


@cacheable
@defines_strategy()
def multi_gbms(
    *,
    initial_value: float,
    num_points: int,
    num_rvs: int,
    params: List,
    scale: List[List],
    shift: List,
    hist_data: Optional[List[List]] = None,
) -> st.SearchStrategy[npt.ArrayLike]:
    """
    Generates instances of ``np.ndarray``. The generated random instances
    are individual runs of a multivariate Monte Carlo sim driven by
    correlated Geometric Brownian motions.

    Args:
        initial_value (float): The initial value of each sim.
        num_points (int): The number of points to generate for each sim.
        num_rvs (int): The number of random variables for each sim.
        params (List): The base parameter arguments (e.g. loc, scale) of the random variables.
        scale (List[List]): The scale matrix to mix random variables via affine transformation.
        shift (List): The shift vector to translate random variables via affine transformation.
        hist_data (Optional[List[List]]): Historical data to fit the random variable params from.

    Returns:
        :class:`hypothesis.strategies.SearchStrategy[numpy.typing.ArrayLike]`
    """
    check_type(float, initial_value, "initial_value")
    check_type(int, num_points, "num_points")
    check_type(int, num_rvs, "num_rvs")
    check_type(list, params, "params")
    check_type(list, scale, "scale")
    check_type(list, shift, "shift")
    if hist_data is not None:
        check_type(list, hist_data, "hist_data")

        # fit to log differences: log(p[i+1] / p[i])
        hist_data = np.diff(np.log(np.asarray(hist_data)), axis=0).tolist()

    from ape_risk.strategies.simulation import MultivariateSimulationStrategy

    strat = MultivariateSimulationStrategy(
        dist_type="norm",
        num_points=num_points,
        num_rvs=num_rvs,
        params=params,
        scale=scale,
        shift=shift,
        hist_data=hist_data,
    )

    def pack(x: npt.ArrayLike) -> npt.ArrayLike:
        return initial_value * np.exp(np.cumsum(x, axis=0))  # axis=0 sums over rows

    return strat.map(pack)

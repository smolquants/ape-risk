from typing import List, Optional

import numpy as np
import numpy.typing as npt
from hypothesis import strategies as st
from hypothesis.internal.conjecture.data import ConjectureData

from ape_risk.stats import MonteCarlo, MultivariateMonteCarlo


class SimulationStrategy(st.SearchStrategy):
    """
    Monte Carlo simulation strategy.
    """

    _mc: MonteCarlo

    def __init__(
        self,
        dist_type: str,
        num_points: int,
        params: List,
        hist_data: Optional[List] = None,
    ):
        # init the monte carlo simulator
        self._mc = MonteCarlo(
            dist_type=dist_type,
            num_points=num_points,
            num_sims=1,  # for each hypothesis run to be a single sim
        )
        self._mc.freeze(params)

        # fit params to historical data if given
        if hist_data is not None:
            if self._mc.num_points > len(hist_data):
                raise ValueError(
                    f"Sample size {self._mc.num_points} not in expected range 0 <= num_points <= {len(hist_data)}"  # noqa: E501
                )

            self._mc.fit(np.asarray(hist_data))

    def do_draw(self, data: ConjectureData) -> npt.ArrayLike:
        """
        Draws a new Monte Carlo simulation using a 32-bit generated seed.

        Args:
            data (:class:`hypothesis.internal.conjecture.data.ConjectureData`):
                The conjecture data for the draw.

        Returns:
            numpy.typing.ArrayLike
        """
        seed = data.draw_bits(32)  # 32 bits is max for seed to np.random.seed
        np.random.seed(
            seed
        )  # TODO: fix since not best practice; see: https://numpy.org/devdocs/reference/random/generated/numpy.random.RandomState.seed.html  # noqa: E501
        return self._mc.sims()


class MultivariateSimulationStrategy(st.SearchStrategy):
    """
    Multivariate Monte Carlo simulation strategy.
    """

    _mmc: MultivariateMonteCarlo

    def __init__(
        self,
        dist_type: str,
        num_points: int,
        num_rvs: int,
        params: List,
        scale: List[List],
        shift: List,
        hist_data: Optional[List[List]] = None,
    ):
        # init the monte carlo simulator
        self._mmc = MultivariateMonteCarlo(
            dist_type=dist_type,
            num_points=num_points,
            num_sims=1,  # for each hypothesis run to be a single sim
            num_rvs=num_rvs,
        )
        self._mmc.freeze(params)
        self._mmc.mix(np.asarray(scale), np.asarray(shift))

        # fit scale, shift to historical data if given
        if hist_data is not None:
            hist_size = np.asarray(hist_data).shape[0]
            if self._mmc.num_points > hist_size:
                raise ValueError(
                    f"Sample size {self._mmc.num_points} not in expected range 0 <= num_points <= {hist_size}"  # noqa: E501
                )

            self._mmc.fit(np.asarray(hist_data))

    def do_draw(self, data: ConjectureData) -> npt.ArrayLike:
        """
        Draws a new multivariate Monte Carlo simulation using a 32-bit generated seed.

        Args:
            data (:class:`hypothesis.internal.conjecture.data.ConjectureData`):
                The conjecture data for the draw.

        Returns:
            numpy.typing.ArrayLike
        """
        seed = data.draw_bits(32)  # 32 bits is max for seed to np.random.seed
        np.random.seed(
            seed
        )  # TODO: fix since not best practice; see: https://numpy.org/devdocs/reference/random/generated/numpy.random.RandomState.seed.html  # noqa: E501
        return self._mmc.sims()

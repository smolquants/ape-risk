from typing import List, Optional

import numpy as np
import numpy.typing as npt
from hypothesis import strategies as st
from hypothesis.internal.conjecture.data import ConjectureData

from ape_risk.stats import MonteCarlo


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

            self._mc.fit(hist_data)

    def do_draw(self, data: ConjectureData) -> npt.ArrayLike:
        seed = data.draw_bits(32)
        np.random.seed(seed)  # TODO: fix this random.seed is deprecated/old
        return self._mc.sims()

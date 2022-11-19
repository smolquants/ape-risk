from typing import Optional

import numpy.typing as npt
from hypothesis import strategies as st
from hypothesis.internal.conjecture.data import ConjectureData

from ape_risk.stats import MonteCarlo


class Simulation(st.SearchStrategy):
    def __init__(
        self,
        dist_type: str,
        num_points: int,
        params: npt.ArrayLike,
        hist_data: Optional[npt.ArrayLike] = None,
    ):
        # init the monte carlo simulator
        self._mc = MonteCarlo(
            dist_type=self.dist_type,
            num_points=self.num_points,
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
        return self._mc.sims()

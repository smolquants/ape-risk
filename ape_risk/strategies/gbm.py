from typing import List, Optional

import numpy as np
import numpy.typing as npt
from hypothesis.internal.conjecture.data import ConjectureData

from ape_risk.strategies.simulation import SimulationStrategy


class GBMStrategy(SimulationStrategy):
    """
    Monte Carlo simulation strategy for Geometric Brownian motion.
    """

    _initial_value: float

    def __init__(
        self, initial_value: float, num_points: int, params: List, hist_data: Optional[List] = None
    ):
        self._initial_value = initial_value
        super().__init__(
            dist_type="norm", num_points=num_points, params=params, hist_data=hist_data
        )

    def do_draw(self, data: ConjectureData) -> npt.ArrayLike:
        # TODO: data.draw()?
        log_values = np.cumsum(super().do_draw(data))
        return self._initial_value * np.exp(log_values)

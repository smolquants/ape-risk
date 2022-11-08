from typing import List

import numpy as np
from scipy import stats  # type: ignore


class MonteCarlo:
    def generate_rvs(
        self, *args: List[int], dist: stats.rv_continuous, num_points: int, num_sims: int
    ) -> np.ndarray:
        """
        Generates iid samples from given distribution for size = (num_points, num_sims).

        Args:
            *args (List[int]): The parameter arguments (e.g. loc, scale) of `dist`.
            dist (:class:`scipy.stats.rv_continuous`): The continuous distribution to sample from.
            num_points (int): The number of points to generate for each sim.
            num_sims (int): The number of sims.
        """
        # TODO: correlation matrix
        # TODO: esimtated density inputs
        return dist.rvs(*args, size=(num_points, num_sims))

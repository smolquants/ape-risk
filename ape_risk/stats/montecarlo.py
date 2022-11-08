from typing import List

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy import stats  # type: ignore


class MonteCarlo(BaseModel):
    """
    Monte carlo simulator.

    Attrs:
        dist_type (str): The continuous distribution to sample from.
        params (List[float]): The parameter arguments (e.g. loc, scale) of `dist`.
        num_points (int): The number of points to generate for each sim.
        num_sims (int): The number of sims.

        TODO: correlation matrix, estimated density
    """

    dist_type: str
    params: List[float]
    num_points: int
    num_sims: int

    @property
    def dist(self) -> stats.rv_continuous:
        d = getattr(stats, self.dist_type)
        if d is None or not isinstance(d, stats.rv_continuous):
            raise Exception(f"dist_type {self.dist_type} not supported")
        return d

    def sims(self) -> np.ndarray:
        """
        Generates iid samples from given distribution for size = (num_points, num_sims).
        """
        return self.dist.rvs(*self.params, size=(self.num_points, self.num_sims))

    def fit(self, data: npt.ArrayLike):
        """
        Fits self.params from given data.
        """
        params = self.dist.fit(data)
        self.params = params

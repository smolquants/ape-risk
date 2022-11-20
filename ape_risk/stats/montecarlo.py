from typing import ClassVar, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, validator
from scipy import stats  # type: ignore


class MonteCarlo(BaseModel):
    """
    Monte carlo simulator.

    Attrs:
        dist_type (str): The continuous distribution to sample from.
        num_points (int): The number of points to generate for each sim.
        num_sims (int): The number of sims.

        TODO: correlation matrix, estimated density
    """

    dist_type: str
    num_points: int
    num_sims: int
    supported_dist_types: ClassVar[Tuple] = (stats.rv_continuous, stats.rv_discrete)

    _rv: Optional[Union[stats.rv_continuous, stats.rv_discrete]] = None

    class Config:
        underscore_attrs_are_private = True

    @validator("dist_type")
    def dist_type_supported(cls, v):
        d = getattr(stats, v)
        assert d is not None and isinstance(
            d, cls.supported_dist_types
        ), f"dist_type {v} not supported"
        return v

    @property
    def dist(self) -> Union[stats.rv_continuous, stats.rv_discrete]:
        """
        The distribution class of the random variable to sample from.
        """
        return getattr(stats, self.dist_type)

    @property
    def rv(self) -> Union[stats.rv_continuous, stats.rv_discrete]:
        """
        The random variable to sample from.
        """
        if self._rv is None:
            raise Exception("dist not frozen with rv params")
        return self._rv

    @property
    def params(self) -> npt.ArrayLike:
        """
        The distributional parameters of the random variable to sample from.
        """
        return np.asarray(self.rv.args)

    def freeze(self, params: npt.ArrayLike):
        """
        Freezes the distribution as a random variable using the given params.

        Args:
            params (npt.ArrayLike): The parameter arguments (e.g. loc, scale) of `rv`.
        """
        self._rv = self.dist(*params)

    def sims(self) -> npt.ArrayLike:
        """
        Generates iid samples from given distribution for size = (num_points, num_sims).

        Returns:
            numpy.typing.ArrayLike
        """
        return self.rv.rvs(size=(self.num_points, self.num_sims))

    def fit(self, data: npt.ArrayLike):
        """
        Fits distribution params then freezes as random variable using the given data.

        Args:
            data (npt.ArrayLike): The data to fit the random variable params from.
        """
        params = self.dist.fit(data)
        self.freeze(params)

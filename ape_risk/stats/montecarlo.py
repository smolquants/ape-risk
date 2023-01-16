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

    def fit(self, data: np.ndarray):
        """
        Fits distribution params then freezes as random variable using the given data.

        Args:
            data (np.ndarray): The data to fit the random variable params from.
        """
        params = self.dist.fit(data)
        self.freeze(params)


class MultivariateMonteCarlo(MonteCarlo):
    """
    Multivariate monte carlo simulator.

    Attrs:
        dist_type (str): The continuous distribution to sample from.
        num_points (int): The number of points to generate for each sim.
        num_sims (int): The number of sims.
        num_rvs (int): The number of rvs to use for sims.
    """

    num_rvs: int

    _scale: Optional[np.ndarray] = None
    _shift: Optional[np.ndarray] = None

    @validator("dist_type")
    def dist_type_supported(cls, v):
        # TODO: support more dists
        assert v == "norm", f"dist_type {v} not supported"
        return v

    @property
    def scale(self) -> np.ndarray:
        if self._scale is None:
            raise Exception("dist not mixed with transform properties")
        return self._scale

    @property
    def shift(self) -> np.ndarray:
        if self._shift is None:
            raise Exception("dist not mixed with transform properties")
        return self._shift

    def mix(self, scale: np.ndarray, shift: np.ndarray):
        """
        Sets the mixing transformation properties to use in generating correlated
        sims from iid random variables.

        Args:
            scale (npt.ArrayLike): The scale matrix.
            shift (npt.ArrayLike): The shift vector.
        """
        # check shapes
        if scale.shape != (self.num_rvs, self.num_rvs):
            raise ValueError(f"Scale matrix is not shape of ({self.num_rvs}, {self.num_rvs})")
        if shift.shape != (self.num_rvs,):
            raise ValueError(f"Shift vector is not shape of (1, {self.num_rvs})")

        self._scale = scale
        self._shift = shift

    def sims(self) -> npt.ArrayLike:
        """
        Generates correlated samples from given distribution for
        size = (num_points, num_sims, num_rvs).

        Returns:
            numpy.typing.ArrayLike
        """
        x = self.rv.rvs(size=(self.num_points, self.num_sims, self.num_rvs))
        return np.einsum("ij,nmj->nmi", self.scale, x) + self.shift

    def fit(self, data: np.ndarray):
        """
        Fits distribution params then freezes as random variable and
        mixes with affine transform using the given data.

        Args:
            data (np.ndarray): The data to fit the random variable params from.
        """
        if data.shape != (data.shape[0], self.num_rvs):
            raise ValueError(f"data is not shape of (_, {self.num_rvs})")

        # TODO: generalize to stable family
        shift = np.mean(data, axis=0)
        C = np.cov(data.T)
        scale = np.linalg.cholesky(C)
        params = (0, 1)  # standard normal

        self.freeze(params)
        self.mix(scale, shift)

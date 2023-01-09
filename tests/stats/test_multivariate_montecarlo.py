import numpy as np
import pytest


def test_scale(mmc):
    # test fails when mix not called yet
    with pytest.raises(Exception):
        _ = mmc.scale

    # test succeeds with manually set private attr
    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    scale = np.linalg.cholesky(C)
    mmc._scale = scale
    mmc_scale = mmc.scale
    np.testing.assert_equal(mmc_scale, scale)


def test_shift(mmc):
    # test fails when mix not called yet
    with pytest.raises(Exception):
        _ = mmc.shift

    # test succeeds with manually set private attr
    shift = np.asarray([0.1, 0.2, 0.3])
    mmc._shift = shift
    mmc_shift = mmc.shift
    np.testing.assert_equal(mmc_shift, shift)


def test_mix(mmc):
    pass


def test_sims(mmc):
    pass


def test_fit(mmc):
    pass

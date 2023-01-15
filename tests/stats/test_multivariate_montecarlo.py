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


def test_mix_sets_properties(mmc):
    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    scale = np.linalg.cholesky(C)
    shift = np.asarray([0.1, 0.2, 0.3])
    mmc.mix(scale, shift)

    np.testing.assert_equal(mmc.scale, scale)
    np.testing.assert_equal(mmc.shift, shift)


def test_mix_fails_when_invalid_scale_shape(mmc):
    C = np.asarray([[1, 0.1], [0.1, 1]])
    scale = np.linalg.cholesky(C)
    shift = np.asarray([0.1, 0.2, 0.3])
    with pytest.raises(Exception):
        mmc.mix(scale, shift)


def test_mix_fails_when_invalid_shift_shape(mmc):
    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    scale = np.linalg.cholesky(C)
    shift = np.asarray([0.1, 0.2])
    with pytest.raises(Exception):
        mmc.mix(scale, shift)


def test_sims(mmc):
    # freeze the dist for a rv to sample from
    params = [0, 0.001]  # loc, scale
    mmc.freeze(params)

    # mix the dist for multiple correlated rvs
    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    scale = np.linalg.cholesky(C)
    shift = np.asarray([0.1, 0.2, 0.3])
    mmc.mix(scale, shift)

    # generate samples and check return shape is correct
    nd_samples = mmc.sims()
    assert isinstance(nd_samples, np.ndarray)
    assert nd_samples.shape == (mmc.num_points, mmc.num_sims, mmc.num_rvs)

    # check shift of all samples (assumes iid)
    # flatten nd array
    size = mmc.num_points * mmc.num_sims
    nd_samples_reshaped = np.reshape(nd_samples, (size, mmc.num_rvs))

    # check mean for shift (since norm)
    # TODO: update when non-norm allowed
    fit_shift = np.mean(nd_samples_reshaped, axis=0)
    np.testing.assert_allclose(fit_shift, mmc.shift, rtol=1e-2)

    # check covariance for scale (since norm)
    # TODO: update when non-norm allowed
    fit_C = np.cov(nd_samples_reshaped.T) / params[1] ** 2
    fit_scale = np.linalg.cholesky(fit_C)
    np.testing.assert_allclose(fit_scale, mmc.scale, rtol=5e-2)


def test_fit(mmc):
    # freeze the dist for a rv to sample from
    params = [0, 0.001]  # loc, scale
    mmc.freeze(params)

    # mix the dist for multiple correlated rvs
    C = np.asarray([[1, 0.1, 0.2], [0.1, 1, 0.1], [0.2, 0.1, 1]])
    scale = np.linalg.cholesky(C)
    shift = np.asarray([0.1, 0.2, 0.3])
    mmc.mix(scale, shift)

    new_params = list(np.asarray(params) * 0.5)  # new params
    new_scale = scale * 0.5
    new_shift = shift * 0.5

    # generate data
    size = mmc.num_points * mmc.num_sims
    x = mmc.dist.rvs(*new_params, size=(size, mmc.num_rvs))

    # affine transform rvs with new scale and shift
    nd_samples = np.einsum("ij,nj->ni", new_scale, x) + new_shift

    # fit shift, scale to data
    mmc.fit(nd_samples)

    # check mmc.params refit to standard
    # TODO: update when non-norm allowed
    np.testing.assert_equal([0, 1], mmc.params)

    # adjust expected scale for initial new_params[1] scale
    new_scale_mixed = new_scale * new_params[1]

    # check mmc.scale, mmc.shift refit to close to new shift, scale
    np.testing.assert_allclose(new_scale_mixed, mmc.scale, rtol=5e-2)
    np.testing.assert_allclose(new_shift, mmc.shift, rtol=1e-2)

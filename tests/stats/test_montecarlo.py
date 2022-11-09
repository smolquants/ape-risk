import numpy as np
import pytest
from scipy import stats  # type: ignore


def test_dist(mc):
    # test succeeds when dist type supported by stats
    assert mc.dist == getattr(stats, mc.dist_type)

    # test fails when not supported by stats
    mc.dist_type = "fake_dist"
    with pytest.raises(Exception):
        _ = mc.dist


def test_rv(mc):
    params = [0.1, 0.001]  # loc, scale

    # test rv throws when not frozen
    with pytest.raises(Exception):
        mc.rv

    # test rv does not throw after frozen and matches expected
    mc.freeze(params)
    rv = mc.dist(*params)
    np.testing.assert_equal(mc.rv.dist._updated_ctor_param(), rv.dist._updated_ctor_param())
    np.testing.assert_equal(mc.rv.args, rv.args)


def test_params(mc):
    # freeze the dist for a rv to sample from
    params = [0.1, 0.001]  # loc, scale
    mc.freeze(params)

    # check params match what was frozen
    np.testing.assert_equal(mc.params, params)


def test_freeze(mc):
    # freeze the dist for a rv to sample from
    params = [0.1, 0.001]  # loc, scale
    mc.freeze(params)

    # expected frozen rv
    rv = mc.dist(*params)

    # check private _rv set
    np.testing.assert_equal(mc._rv.dist._updated_ctor_param(), rv.dist._updated_ctor_param())
    np.testing.assert_equal(mc._rv.args, rv.args)


def test_sims(mc):
    # freeze the dist for a rv to sample from
    params = [0.1, 0.001]  # loc, scale
    mc.freeze(params)

    # generate samples and check return shape is correct
    nd_samples = mc.sims()
    assert isinstance(nd_samples, np.ndarray)
    assert nd_samples.shape == (mc.num_points, mc.num_sims)

    # check loc, scale of all samples (assumes iid)
    # flatten nd array
    size = mc.num_points * mc.num_sims
    nd_samples_reshaped = np.reshape(nd_samples, size)

    # fit and check params close
    fit_params = mc.dist.fit(nd_samples_reshaped)
    np.testing.assert_allclose(fit_params, mc.params, rtol=1e-2)


def test_fit(mc):
    # freeze the dist for a rv to sample from
    params = [0.1, 0.001]  # loc, scale
    mc.freeze(params)

    new_params = list(np.asarray(params) * 0.5)  # new params

    # generate data
    size = mc.num_points * mc.num_sims
    nd_samples = mc.dist.rvs(*new_params, size=size)

    # fit mc.params to data
    mc.fit(nd_samples)

    # check mc.params refit to close to new params
    np.testing.assert_allclose(new_params, mc.params, rtol=1e-2)

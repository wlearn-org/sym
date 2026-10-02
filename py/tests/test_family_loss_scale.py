"""Target-unit policy, fitted on training rows and persisted in the C payload."""

import numpy as np
import pytest
from wlearn_sym import SymbolicClassifier, SymbolicRegressor


def variance(y):
    # Match the canonical engine's float32 targets and stable population moment.
    mean = total = 0.0
    for n, value in enumerate(np.asarray(y, dtype=np.float32), 1):
        delta = float(value) - mean
        mean += delta / n
        total += delta * (float(value) - mean)
    return max(0.0, total / len(y))


def params(**extra):
    return dict(
        strategy="family",
        population=16,
        eliteCount=4,
        generations=8,
        terms=3,
        polishPasses=2,
        localRefineInterval=2,
        localRefineCount=2,
        seed=31,
        **extra,
    )


@pytest.mark.parametrize("method", ["coordinate", "lm"])
def test_variance_policy_matches_explicit_absolute_settings(method):
    X = np.random.default_rng(610).uniform(-2, 2, (80, 3))
    y = 0.001 * (np.cos(X[:, 0]) + X[:, 1] * X[:, 2])
    v = variance(y)
    a = SymbolicRegressor(
        params(
            polishMethod=method,
            lossScale="target-variance",
            complexityPenalty=0.01,
            tol=0.001,
            earlyStopRounds=2,
        )
    )
    b = SymbolicRegressor(
        params(
            polishMethod=method,
            complexityPenalty=0.01 * v,
            tol=0.001 * v,
            earlyStopRounds=2,
        )
    )
    try:
        a.fit(X, y)
        b.fit(X, y)
        assert a.formula() == b.formula()
        np.testing.assert_array_equal(a.predict(X), b.predict(X))
        loaded = SymbolicRegressor.load(a.save())
        try:
            assert loaded.formula() == a.formula()
            np.testing.assert_array_equal(loaded.predict(X), a.predict(X))
        finally:
            loaded.dispose()
    finally:
        a.dispose()
        b.dispose()


@pytest.mark.parametrize("target", [np.zeros(32), np.full(32, 7.0)])
def test_constant_targets_use_zero_scale(target):
    X = np.arange(32).reshape(-1, 1)
    m = SymbolicRegressor(params(loss_scale="target-variance"))
    try:
        m.fit(X, target)
        f = m.formula()
        assert f["objective"] == f["validLoss"] == 0
        np.testing.assert_array_equal(m.predict(X), target)
    finally:
        m.dispose()


@pytest.mark.parametrize("option", ["bogus", True, 1])
def test_invalid_loss_scale(option):
    m = SymbolicRegressor(params(lossScale=option))
    try:
        with pytest.raises(ValueError, match="loss scale"):
            m.fit(np.arange(32).reshape(-1, 1), np.arange(32))
    finally:
        m.dispose()


def test_tree_rejects_loss_scale():
    with pytest.raises(ValueError, match="family"):
        SymbolicRegressor(dict(lossScale="target-variance"))


def test_classifier_rejects_target_variance():
    m = SymbolicClassifier(params(lossScale="target-variance"))
    try:
        with pytest.raises(ValueError, match="regression"):
            m.fit(np.arange(32).reshape(-1, 1), np.arange(32) % 2)
    finally:
        m.dispose()


def test_training_only_scale_and_repeatable_c_configuration():
    import ctypes as ct
    import struct
    from wlearn_sym._ffi import get_lib, _DP
    from wlearn_sym._family_c import _config

    lib = get_lib()
    lib.sym_family_search_set_loss_scale.argtypes = [ct.c_void_p, ct.c_int]
    lib.sym_family_search_set_loss_scale.restype = ct.c_int
    lib.sym_family_search_run.argtypes = [ct.c_void_p]
    lib.sym_family_search_run.restype = ct.c_void_p
    X = np.random.default_rng(74).normal(size=(32, 2))
    y = np.arange(32, dtype=float) * 0.001
    config = _config(params(validationFraction=0.25, lossScale="target-variance"))

    def ptr(a):
        return a.ctypes.data_as(_DP)

    def new(target):
        state = lib.wl_sym_family_search_new(
            ptr(X), 32, 2, ptr(target), 0, 0, ptr(config), len(config), 4, 16
        )
        assert state
        return state

    state = new(y)
    try:
        assert lib.wl_sym_family_search_propose(state) == 1
        data = np.empty(65)
        assert lib.sym_family_search_data(state, ptr(data), 65) == 0
        assert lib.sym_family_search_set_loss_scale(state, 1) == -1
        mask = data[32:64].astype(bool)
    finally:
        lib.sym_family_search_free(state)
    expected = variance(y[mask]) * 1e-5
    changed = y.copy()
    changed[~mask] = 1e5
    for target in (y, changed):
        state = new(target)
        model = None
        try:
            for mode in (1, 1, 0, 1):
                assert lib.sym_family_search_set_loss_scale(state, mode) == 0
            assert lib.sym_family_search_set_loss_scale(state, 7) == -1
            model = lib.sym_family_search_run(state)
            assert model
            out, size = ct.c_void_p(), ct.c_int()
            assert lib.sym_family_save(model, ct.byref(out), ct.byref(size)) == 0
            try:
                penalty = struct.unpack_from("<d", ct.string_at(out, size.value), 36)[0]
                assert penalty == expected
            finally:
                lib.wl_sym_free_buffer(out)
        finally:
            if model:
                lib.sym_family_free(model)
            lib.sym_family_search_free(state)


@pytest.mark.parametrize("field", ["complexityPenalty", "tol"])
def test_scaled_setting_overflow_rejected(field):
    m = SymbolicRegressor(params(lossScale="target-variance", **{field: 1e308}))
    try:
        with pytest.raises(ValueError, match="overflow"):
            m.fit(np.arange(32).reshape(-1, 1), np.arange(32) * 1e10)
    finally:
        m.dispose()


@pytest.mark.parametrize("scale", [2**-20, 1, 2**20])
def test_relative_policy_preserves_target_unit_changes(scale):
    X = np.random.default_rng(92).normal(size=(80, 3))
    y = np.cos(X[:, 0]) + X[:, 1] * X[:, 2]
    p = params(
        lossScale="target-variance", polishMethod="lm", tol=0.001, earlyStopRounds=2
    )
    a, b = SymbolicRegressor(p), SymbolicRegressor(p)
    try:
        a.fit(X, y)
        b.fit(X, y * scale)
        # Powers of two avoid adding arbitrary target-rounding differences.
        np.testing.assert_allclose(
            a.predict(X), b.predict(X) / scale, rtol=1e-12, atol=1e-12
        )
        assert a.formula()["complexity"] == b.formula()["complexity"]
    finally:
        a.dispose()
        b.dispose()

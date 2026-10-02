import numpy as np
import pytest
from wlearn_sym import SymbolicRegressor


def test_scaled_lm_escapes_raw_constant_bounds_and_roundtrips():
    x = np.linspace(-1, 1, 160).reshape(-1, 1) * 0.01
    y = np.sin(230 * x[:, 0] + 0.4)
    m = SymbolicRegressor(
        dict(
            strategy="family",
            operatorSet=["sin"],
            terms=1,
            population=16,
            eliteCount=4,
            generations=3,
            seed=11,
            validationFraction=0.2,
            scaleAware=True,
            polishMethod="lm",
            polishPasses=8,
            localRefineInterval=1,
            localRefineCount=2,
        )
    )
    loaded = None
    try:
        m.fit(x, y)
        assert np.mean((m.predict(x) - y) ** 2) < 1e-4
        loaded = SymbolicRegressor.load(m.save())
        np.testing.assert_array_equal(m.predict(x), loaded.predict(x))
        assert abs(m.formula()["terms"][0]["p0"]) > 16
    finally:
        m.dispose()
        if loaded is not None:
            loaded.dispose()


@pytest.mark.parametrize(
    "options",
    [
        dict(scaleAware="yes"),
        dict(polishMethod="bogus"),
        dict(polishMethod="lm", polishPasses=0),
        dict(polishMethod="lm", polishBatchSize=4),
    ],
)
def test_invalid_optimizer_options(options):
    m = SymbolicRegressor({"strategy": "family", "polishPasses": 2, **options})
    try:
        with pytest.raises(ValueError):
            m.fit(np.arange(20).reshape(10, 2), np.arange(10))
    finally:
        m.dispose()


@pytest.mark.parametrize("offset,collinear", [(0, False), (1e6, False), (0, True)])
def test_damped_qr_step_matches_numpy(offset, collinear):
    import ctypes as ct
    from wlearn_sym._ffi import get_lib

    lib = get_lib()
    dp = ct.POINTER(ct.c_double)
    fn = lib.sym_family_lm_step
    fn.argtypes = [dp, ct.c_int, ct.c_int, dp, ct.c_double, dp]
    fn.restype = ct.c_int
    rng = np.random.default_rng(55)
    residual = rng.normal(size=80) + offset
    jac = rng.normal(size=(80, 6))
    if collinear:
        jac[:, 1] = jac[:, 0] + 1e-9 * jac[:, 1]
    h = np.full(6, 0.002)
    samples = np.ascontiguousarray(
        np.column_stack([residual, residual[:, None] + jac * h]).T
    )
    actual = np.empty(6)

    def ptr(x):
        return x.ctypes.data_as(dp)

    assert fn(ptr(samples), 80, 6, ptr(h), 0.001, ptr(actual)) == 0
    numeric = (samples[1:].T - samples[0, :, None]) / h
    scale = np.linalg.norm(numeric, axis=0)
    expected = (
        np.linalg.lstsq(
            np.vstack([numeric / scale, np.sqrt(0.001) * np.eye(6)]),
            np.r_[-residual, np.zeros(6)],
            rcond=None,
        )[0]
        / scale
    )
    expected *= min(1, 2 / np.linalg.norm(expected))
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def test_scaling_statistics_exclude_validation_rows():
    import ctypes as ct
    from wlearn_sym._ffi import get_lib
    from wlearn_sym._family_c import _config

    lib = get_lib()
    dp = ct.POINTER(ct.c_double)

    def ptr(x):
        return x.ctypes.data_as(dp)

    X = np.linspace(-0.01, 0.01, 60).reshape(-1, 1)
    y = np.sin(230 * X[:, 0])
    config = _config(
        dict(
            population=8,
            eliteCount=2,
            terms=2,
            generations=1,
            validationFraction=0.25,
            scaleAware=True,
        )
    )

    def descriptors(X, y):
        s = lib.wl_sym_family_search_new(
            ptr(X), 60, 1, ptr(y), 0, 0, ptr(config), len(config), 8, 60
        )
        assert s
        try:
            assert lib.wl_sym_family_search_propose(s) == 1
            d = np.empty((8, 2, 5))
            data = np.empty(121)
            assert lib.wl_sym_family_batch_descriptors(s, ptr(d), d.size) == 0
            assert lib.sym_family_search_data(s, ptr(data), data.size) == 0
            return d, data[60:120]
        finally:
            lib.sym_family_search_free(s)

    original, mask = descriptors(X, y)
    X[mask == 0] = 1e6
    y[mask == 0] = -1e6
    changed, mask2 = descriptors(X, y)
    np.testing.assert_array_equal(original, changed)
    np.testing.assert_array_equal(mask, mask2)


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("offset", [0, 1e6])
def test_lm_offset_collinear_and_zero_columns(hierarchical, offset):
    rng = np.random.default_rng(319)
    x = rng.normal(size=64)
    X = np.column_stack(
        [x + offset, x + offset + 1e-7 * rng.normal(size=64), np.zeros(64)]
    )
    y = x * 0.3 + np.sin(x)
    m = SymbolicRegressor(
        dict(
            strategy="family",
            population=8,
            eliteCount=2,
            generations=2,
            terms=3,
            polishPasses=2,
            polishMethod="lm",
            scaleAware=True,
            hierarchical=hierarchical,
        )
    )
    try:
        m.fit(X, y)
        assert np.isfinite(m.predict(X)).all()
        saved = m.save()
        loaded = SymbolicRegressor.load(saved)
        try:
            np.testing.assert_array_equal(m.predict(X), loaded.predict(X))
        finally:
            loaded.dispose()
    finally:
        m.dispose()


def test_lm_reduced_residual_validation_is_atomic():
    import ctypes as ct
    from wlearn_sym._ffi import get_lib
    from wlearn_sym._family_c import _config

    lib = get_lib()

    def ptr(x):
        return x.ctypes.data_as(ct.POINTER(ct.c_double))

    X = np.linspace(-1, 1, 48).reshape(-1, 1)
    y = np.sin(X[:, 0])
    config = _config(
        dict(
            population=4,
            eliteCount=1,
            generations=1,
            terms=1,
            operatorSet=["sin"],
            polishPasses=2,
            polishMethod="lm",
            validationFraction=0.25,
        )
    )
    state = lib.wl_sym_family_search_new(
        ptr(X), 48, 1, ptr(y), 0, 0, ptr(config), len(config), 2, 48
    )
    assert state
    try:
        while lib.wl_sym_family_search_propose(state) > 0:
            identity = lib.wl_sym_family_batch_id(state)
            shape = (ct.c_int * 8)()
            assert lib.wl_sym_family_batch_shape(state, shape, 8) == 0
            _, count, rows, _, terms, _, _, _ = shape
            if lib.sym_family_search_residual_count(state):
                data = np.empty(97)
                assert lib.sym_family_search_data(state, ptr(data), 97) == 0
                packed = np.zeros((count, terms + 3 + 48))
                assert (
                    lib.sym_family_search_accept_results(
                        state, identity, ptr(packed), packed.size - 1
                    )
                    == -1
                )
                packed[0, terms + 3 + np.flatnonzero(data[48:96] == 0)[0]] = 1
                assert (
                    lib.sym_family_search_accept_results(
                        state, identity, ptr(packed), packed.size
                    )
                    == -1
                )
                assert lib.wl_sym_family_batch_id(state) == identity
                break
            values = np.empty((count, rows, terms))
            assert (
                lib.sym_family_search_score(state, identity, ptr(values), values.size)
                == 0
            )
            assert (
                lib.sym_family_search_accept(state, identity, ptr(values), values.size)
                == 0
            )
        else:
            pytest.fail("search did not request LM residuals")
    finally:
        lib.sym_family_search_free(state)


@pytest.mark.parametrize("option", [{"scaleAware": True}, {"polishMethod": "lm"}])
def test_tree_rejects_family_optimizer_options(option):
    with pytest.raises(ValueError, match="family"):
        SymbolicRegressor(option)

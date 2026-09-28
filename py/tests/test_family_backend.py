"""Optional Polygrad backend against the canonical C search, including polish."""

import numpy as np
import pytest
from wlearn_sym import SymbolicRegressor, SymbolicClassifier

pytest.importorskip("polygrad")


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("classes", [0, 2, 3])
def test_shared_search_polygrad(classes, hierarchical):
    rng = np.random.default_rng(319)
    X = rng.uniform(-1, 1, (64, 3))
    y = np.arange(64) % classes if classes else np.sin(X[:, 0]) + X[:, 1] * X[:, 2]
    cls = SymbolicClassifier if classes else SymbolicRegressor
    params = dict(
        strategy="family",
        population=16,
        eliteCount=4,
        generations=2,
        terms=3,
        polishPasses=2,
        polishBatchSize=20,
        seed=11,
        validationFraction=0.2,
        hierarchical=hierarchical,
    )
    c, pg = (
        cls(params),
        cls(
            {
                **params,
                "backend": "polygrad",
                "batchSize": 16,
            }
        ),
    )
    try:
        c.fit(X, y)
        pg.fit(X, y)
        np.testing.assert_allclose(pg.predict(X), c.predict(X), atol=1e-5, rtol=1e-5)
        if classes:
            np.testing.assert_allclose(
                pg.predict_proba(X), c.predict_proba(X), atol=1e-5, rtol=1e-5
            )
        np.testing.assert_allclose(
            pg.predict_polygrad(X), pg.predict(X), atol=1e-5, rtol=1e-5
        )
        for head in range(classes if classes > 2 else 1):
            report = pg.refine_polygrad(X, y, index=head, epochs=2)
            assert report["status"] == "ok"
            assert (
                not report["committed"]
                or report["afterObjective"] <= report["beforeObjective"] + 1e-10
            )
        saved = pg.save()
        assert saved[:4] == b"WLRN"
        loaded = cls.load(saved)
        try:
            np.testing.assert_array_equal(loaded.predict(X), pg.predict(X))
        finally:
            loaded.dispose()
    finally:
        c.dispose()
        pg.dispose()


def test_family_gradient_refinement():
    X = np.column_stack([np.linspace(-1, 1, 64), np.sin(np.arange(64))])
    y = np.sin(2.3 * X[:, 0] + 0.4) + 0.1 * X[:, 1]
    m = SymbolicRegressor(
        dict(
            strategy="family",
            population=16,
            generations=2,
            terms=3,
            seed=8,
            validationFraction=0.2,
        )
    )
    try:
        m.fit(X, y)
        report = m.refine_polygrad(X, y, epochs=8, lr=0.01, polygrad={"device": "CPU"})
        assert report["status"] == "ok"
        assert len(report["history"]) == 8
        if report["committed"]:
            assert report["afterObjective"] <= report["beforeObjective"] + 1e-10
        loaded = SymbolicRegressor.load(m.save())
        try:
            np.testing.assert_array_equal(m.predict(X), loaded.predict(X))
        finally:
            loaded.dispose()
    finally:
        m.dispose()


def test_refinement_never_trains_on_validation_labels(monkeypatch):
    from wlearn_sym import _family_refine
    from wlearn_sym._ffi import get_lib, _DP

    X = np.column_stack([np.linspace(-1, 1, 48), np.sin(np.arange(48))])
    y = np.sin(2.1 * X[:, 0]) + 0.2 * X[:, 1]
    model = SymbolicRegressor(
        dict(
            strategy="family",
            terms=2,
            population=12,
            eliteCount=2,
            generations=2,
            seed=9,
            validationFraction=0.25,
        )
    ).fit(X, y)
    other = SymbolicRegressor.load(model.save())
    data = np.empty(2 * len(y) + 1)
    y = np.ascontiguousarray(y)
    assert (
        get_lib().sym_family_refine_data(
            model._family_engine.handle,
            0,
            y.ctypes.data_as(_DP),
            len(y),
            0.25,
            9,
            data.ctypes.data_as(_DP),
            len(data),
        )
        == 0
    )
    changed = y.copy()
    changed[data[len(y) : 2 * len(y)] == 0] += 100
    calls = []
    original = _family_refine.refine

    def capture(candidate, x, target, **kwargs):
        result = original(candidate, x, target, **kwargs)
        calls.append((x.copy(), target.copy(), result[0].copy(), result[1]))
        return result

    monkeypatch.setattr(_family_refine, "refine", capture)
    try:
        model.refine_polygrad(X, y, epochs=2, polygrad={"device": "CPU"})
        other.refine_polygrad(X, changed, epochs=2, polygrad={"device": "CPU"})
        for a, b in zip(calls[0], calls[1]):
            np.testing.assert_array_equal(a, b)
    finally:
        model.dispose()
        other.dispose()


def test_supplied_runtime_survives_fit_predict_refine_and_dispose():
    import polygrad

    pg = polygrad.create()
    X = np.column_stack([np.linspace(-1, 1, 32), np.sin(np.arange(32))])
    y = X[:, 0] + X[:, 1]
    try:
        for _ in range(2):
            model = SymbolicRegressor(
                dict(
                    strategy="family",
                    backend="polygrad",
                    polygrad=pg,
                    batchSize=8,
                    population=8,
                    eliteCount=2,
                    generations=2,
                    terms=2,
                    operatorSet="basic",
                )
            )
            try:
                model.fit(X, y)
                np.testing.assert_allclose(
                    model.predict_polygrad(X), model.predict(X), atol=1e-5
                )
                model.refine_polygrad(X, y, epochs=2)
            finally:
                model.dispose()
            probe = pg.Tensor([3.0])
            try:
                np.testing.assert_array_equal(probe.numpy(), [3.0])
            finally:
                probe.dispose()
    finally:
        pg.dispose()


def test_polygrad_device_environment_is_not_overridden(monkeypatch):
    import subprocess
    import sys

    # Polygrad caches its default device; select the environment before import.
    monkeypatch.setenv("POLY_DEV", "invalid-sym-test-device")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from wlearn_sym import SymbolicRegressor
model = SymbolicRegressor(dict(strategy="family", backend="polygrad",
    population=8, eliteCount=2, generations=1, terms=2, batchSize=8))
try:
    model.fit([[0], [1], [2], [3]], [0, 1, 2, 3])
finally:
    model.dispose()
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    assert "invalid-sym-test-device" in result.stderr.lower(), result.stderr

import numpy as np
import pytest
from wlearn import load
from wlearn.pipeline import Pipeline
from wlearn_sym import SymbolicRegressor, SymbolicClassifier, FormulaTransformer
from wlearn_sym._family_c import CFamilyEngine
import json
from pathlib import Path

LEGACY = json.loads(
    (Path(__file__).parents[2] / "test/fixtures/legacy-family.json").read_text()
)

PARAMS = dict(
    strategy="family",
    backend="c",
    population=24,
    generations=4,
    terms=3,
    eliteCount=4,
    frontierSize=4,
    islands=3,
    validationFraction=0.2,
    seed=123,
    operatorSet="basic",
    polishPasses=2,
)


def data():
    i = np.arange(60)
    X = np.column_stack([(i - 30) / 10, np.sin(i * 0.47)])
    return X, 2 * X[:, 0] + 0.3 * X[:, 1] + 1


@pytest.mark.parametrize(
    "params",
    [
        dict(strategy="bad"),
        dict(backend="bad"),
        dict(strategy="tree", backend="polygrad"),
        dict(engine="c", strategy="family"),
    ],
)
def test_strategy_rejection(params):
    with pytest.raises(ValueError):
        SymbolicRegressor(params)


def test_family_lifecycle_and_existing_payload():
    X, y = data()
    with pytest.raises(ValueError):
        FormulaTransformer(PARAMS)
    m = SymbolicRegressor(PARAMS)
    restored = None
    try:
        assert m.fit(X, y) is m
        assert m.score(X, y) > 0.98
        for index in [0.5, -1, 2**32, np.nan, True]:
            with pytest.raises(ValueError, match="invalid family head index"):
                m.refine_polygrad(X, y, index=index, epochs=1)
        pred = m.predict(X)
        assert m.capabilities["polygradRefinement"]
        restored = load(m.save())
        np.testing.assert_array_equal(restored.predict(X), pred)
        with pytest.raises(ValueError):
            m.set_params(backend="bogus")
        m.set_params(terms=0)
        with pytest.raises(ValueError):
            m.fit(X, y)
        np.testing.assert_array_equal(m.predict(X), pred)
        m.set_params(terms=3)
        state = m._family_engine.to_state()
        for patch in ({"nFeatures": 2**32 + 2}, {"fitted": False}, {"best": None}):
            with pytest.raises(ValueError):
                CFamilyEngine.from_state({**state, **patch})
        for key in (
            "hierarchical",
            "migrationInterval",
            "localRefineCount",
            "scoreMode",
            "jit",
            "stackSummaries",
            "evalCacheSize",
        ):
            m.set_params(**{key: 1})
            with pytest.raises(ValueError):
                m.fit(X, y)
            m.set_params(**{key: None})
        imported = CFamilyEngine.from_state(state)
        np.testing.assert_array_equal(imported.predict(X), pred)
        imported.dispose()
        pipe = Pipeline([("family", restored)])
        pipe.fit(X, y)
        loaded = load(pipe.save())
        try:
            np.testing.assert_array_equal(loaded.predict(X), pipe.predict(X))
        finally:
            loaded.dispose()
    finally:
        m.dispose()
        if restored:
            restored.dispose()


@pytest.mark.parametrize("count", [2, 3])
def test_family_class_order(count):
    X, _ = data()
    classes = [91, -7, 123][:count]
    y = np.asarray(classes)[np.arange(len(X)) % count]
    m = SymbolicClassifier({**PARAMS, "classes": classes})
    restored = None
    try:
        m.fit(X, y)
        assert m.classes == classes
        np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1, atol=1e-12)
        restored = load(m.save())
        np.testing.assert_array_equal(restored.predict_proba(X), m.predict_proba(X))
    finally:
        m.dispose()
        if restored:
            restored.dispose()


@pytest.mark.parametrize("op", range(10))
def test_protected_operator_parity(op):
    # Exercise family protection constants, not the tree evaluator's semantics.
    X = np.array([[0, 0], [1e-8, -1e-8], [-20, 1e-6], [20, -1e-6], [3, 2]])
    fixture = LEGACY["operators"][op]
    native = CFamilyEngine.from_state(fixture["state"])
    try:
        np.testing.assert_allclose(
            native.predict(X), fixture["prediction"], rtol=1e-7, atol=1e-7
        )
    finally:
        native.dispose()


@pytest.mark.parametrize("fixture", LEGACY["bundles"])
def test_legacy_bundle_without_frontend_controller(fixture):
    cls = SymbolicClassifier if fixture["classes"] else SymbolicRegressor
    model = cls.load(Path(__file__).parents[2] / "test/fixtures" / fixture["file"])
    restored = None
    try:
        np.testing.assert_allclose(
            model.predict(fixture["X"]), fixture["prediction"], atol=1e-6, rtol=1e-6
        )
        if fixture["classes"]:
            np.testing.assert_allclose(
                model.predict_proba(fixture["X"]),
                fixture["probability"],
                atol=1e-6,
                rtol=1e-6,
            )
        restored = cls.load(model.save())
        np.testing.assert_array_equal(
            model.predict(fixture["X"]), restored.predict(fixture["X"])
        )
    finally:
        model.dispose()
        if restored:
            restored.dispose()


@pytest.mark.parametrize("offset", [0, 1e6])
def test_ridge_matches_augmented_least_squares(offset):
    x = np.arange(60, dtype=float) / 8
    X = np.column_stack([offset + x, 2 * (offset + x), np.ones(60)])
    y = (2 * x + np.sin(x)).astype(np.float32).astype(float)
    params = {**PARAMS, "operatorSet": ["add"], "validationFraction": 0, "ridge": 0.01}
    m = SymbolicRegressor(params).fit(X, y)
    try:
        c = m._family_engine.to_state()["best"]
        X32 = X.astype(np.float32).astype(float)
        phi = np.column_stack([X32[:, a] + X32[:, b] for a, b in zip(c["fa"], c["fb"])])
        mean = phi.mean(axis=0)
        centered = phi - mean
        augmented = np.vstack(
            [centered, np.sqrt(len(y) * params["ridge"]) * np.eye(c["terms"])]
        )
        rhs = np.concatenate([y - y.mean(), np.zeros(c["terms"])])
        coef = (
            np.linalg.lstsq(augmented, rhs, rcond=None)[0]
            .astype(np.float32)
            .astype(float)
        )
        np.testing.assert_allclose(c["coef"], coef, rtol=1e-5, atol=1e-6)
        expected = ((phi - mean) @ np.asarray(c["coef"]) + y.mean()).astype(np.float32)
        np.testing.assert_allclose(m.predict(X), expected, rtol=1e-6, atol=1e-6)
    finally:
        m.dispose()


def test_versioned_family_artifact_validation_and_legacy_reader():
    import json
    import struct
    from wlearn.bundle import decode_bundle, encode_bundle

    X, y = data()
    model = SymbolicRegressor(PARAMS).fit(X, y)
    try:
        manifest, toc, blobs = decode_bundle(model.save())
        assert manifest["typeId"] == "wlearn.sym.regressor@2"
        assert model.get_params().get("immigrantRate", 0) == 0
        assert 0 in model.default_search_space()["immigrantRate"]["values"]
        raw = bytes(blobs)

        def bundle(payload, meta=manifest):
            return encode_bundle(
                meta,
                [{"id": "model", "mediaType": toc[0]["mediaType"], "data": payload}],
            )

        # Valid hashes do not make malformed inner payloads valid. First term is
        # arithmetic: its p0 must be one, and complexity charges only active genes.
        for offset, value in (
            (48 + 3 * 8, 2.0),
            (48 + (6 * 3 + 3) * 8, 99.0),
            (48 + 5 * 8, 0.123456789),
        ):
            damaged = bytearray(raw)
            struct.pack_into("<d", damaged, offset, value)
            with pytest.raises(ValueError):
                SymbolicRegressor.load(bundle(damaged))
        for damaged in (raw[:-1], raw + b"\0"):
            with pytest.raises(ValueError):
                SymbolicRegressor.load(bundle(damaged))
        for patch in ({"nFeatures": 3}, {"nClasses": 2}, {"classes": [0, 1]}):
            with pytest.raises(ValueError):
                SymbolicRegressor.load(
                    bundle(
                        raw, {**manifest, "metadata": {**manifest["metadata"], **patch}}
                    )
                )
        legacy = {**manifest, "typeId": "wlearn.sym.regressor@1"}
        state = model._family_engine.to_state()
        old = encode_bundle(
            legacy,
            [
                {
                    "id": "model",
                    "mediaType": "application/vnd.wlearn.sym.pg-family+json",
                    "data": json.dumps(state).encode(),
                }
            ],
        )
        restored = SymbolicRegressor.load(old)
        try:
            np.testing.assert_array_equal(restored.predict(X), model.predict(X))
            assert decode_bundle(restored.save())[0]["typeId"] == legacy["typeId"]
        finally:
            restored.dispose()
    finally:
        model.dispose()


def test_hierarchical_formula_text_is_bounded():
    from wlearn_sym._family_format import _formula_text, _candidate_from_plain

    n = 32
    c = dict(
        terms=n,
        fa=list(range(n)),
        fb=list(range(n)),
        op=[0] * n,
        p0=[1] * n,
        p1=[0] * n,
        coef=[1] * n,
        bias=0,
    )
    text = _formula_text(_candidate_from_plain(c), ["x0"])
    assert "t31 = (t30+t30)" in text
    assert len(text) < 5000

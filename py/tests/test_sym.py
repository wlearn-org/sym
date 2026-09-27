from pathlib import Path

import numpy as np

from wlearn.registry import load
from wlearn_sym import (
    FormulaTransformer,
    FormulaVerifier,
    SymbolicClassifier,
    SymbolicRegressor,
    refine_formula_polygrad,
)


SMALL = {
    "population": 80,
    "generations": 45,
    "maxNodes": 15,
    "maxDepth": 4,
    "frontierSize": 8,
    "eliteCount": 4,
    "tournamentSize": 4,
    "validationFraction": 0.2,
    "complexityPenalty": 0.0001,
    "operatorSet": "basic",
}

PG_SMALL = {
    "engine": "pg-family",
    "population": 12,
    "generations": 1,
    "terms": 3,
    "eliteCount": 3,
    "frontierSize": 4,
    "seed": 20260705,
}


def make_regression(n=80):
    X = np.zeros((n, 2), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x0 = -2 + 4 * i / (n - 1)
        x1 = np.sin(i * 0.37)
        X[i] = [x0, x1]
        y[i] = 2 * x0 - 0.5 * x1 + 1
    return X, y


def make_classification(n=90):
    X = np.zeros((n, 2), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x0 = -2 + 4 * i / (n - 1)
        x1 = np.cos(i * 0.19)
        X[i] = [x0, x1]
        y[i] = 10 if x0 + 0.25 * x1 > 0 else 20
    return X, y


def test_regressor_bundle_path_registry_and_polygrad(tmp_path):
    X, y = make_regression()
    model = SymbolicRegressor({**SMALL, "seed": 123}).fit(X, y)
    try:
        assert model.n_features == 2
        assert model.score(X, y) > 0.85
        assert "x" in model.formula(format="text")
        frontier = model.frontier()
        assert frontier
        assert all(frontier[i]["objective"] >= frontier[i - 1]["objective"] - 1e-12 for i in range(1, len(frontier)))

        bundle = model.save()
        loaded = SymbolicRegressor.load(bundle)
        try:
            np.testing.assert_allclose(loaded.predict(X), model.predict(X), atol=1e-12)
        finally:
            loaded.dispose()

        path = tmp_path / "sym-reg.wlrn"
        assert model.save(path) == path.read_bytes()
        loaded_path = SymbolicRegressor.load(Path(path))
        try:
            np.testing.assert_allclose(loaded_path.predict(X), model.predict(X), atol=1e-12)
        finally:
            loaded_path.dispose()

        loaded_registry = load(path)
        try:
            assert loaded_registry.score(X, y) > 0.85
        finally:
            loaded_registry.dispose()

        try:
            poly = model.predict_polygrad(X)
        except ModuleNotFoundError as exc:
            if exc.name != "polygrad":
                raise
        else:
            np.testing.assert_allclose(poly, model.predict(X), atol=1e-4)

        try:
            import polygrad
        except ModuleNotFoundError:
            pass
        else:
            pg = polygrad.create(device="cpu")
            try:
                poly = model.predict_polygrad(X, polygrad=pg)
                np.testing.assert_allclose(poly, model.predict(X), atol=1e-4)
            finally:
                pg.dispose()
    finally:
        model.dispose()


def test_pg_family_regressor_bundle_and_registry(tmp_path):
    X, y = make_regression(36)
    try:
        model = SymbolicRegressor(PG_SMALL).fit(X, y)
    except ModuleNotFoundError as exc:
        if exc.name != "polygrad":
            raise
        return
    try:
        pred = model.predict(X)
        assert pred.shape == (X.shape[0],)
        assert np.isfinite(pred).all()
        assert "x" in model.formula(format="text")
        assert model.formula()["kind"] == "sym.pg-family.formula@1"
        assert model.frontier()

        bundle = model.save()
        loaded = SymbolicRegressor.load(bundle)
        try:
            np.testing.assert_allclose(loaded.predict(X), pred, atol=1e-6)
        finally:
            loaded.dispose()

        path = tmp_path / "sym-pg-reg.wlrn"
        assert model.save(path) == path.read_bytes()
        loaded_path = load(path)
        try:
            np.testing.assert_allclose(loaded_path.predict(X), pred, atol=1e-6)
        finally:
            loaded_path.dispose()
    finally:
        model.dispose()


def test_pg_family_operator_sets_and_stacked_summaries_persist():
    X, y = make_regression(36)
    try:
        model = SymbolicRegressor({
            **PG_SMALL,
            "operators": ["add", "sub", "mul"],
            "stackSummaries": True,
            "generations": 2,
            "seed": 20260707,
        }).fit(X, y)
    except ModuleNotFoundError as exc:
        if exc.name != "polygrad":
            raise
        return
    try:
        formula = model.formula()
        assert all(term["op"] in {"add", "sub", "mul"} for term in formula["terms"])
        timings = model._family_engine.stats["timings"]
        assert timings[0]["fitCount"] == PG_SMALL["population"]
        assert any(item["fitCount"] < PG_SMALL["population"] for item in timings)
        loaded = SymbolicRegressor.load(model.save())
        try:
            params = loaded.get_params()
            assert params["operators"] == ["add", "sub", "mul"]
            assert params["stackSummaries"] is True
            assert "_opIds" not in params
        finally:
            loaded.dispose()
    finally:
        model.dispose()


def test_pg_family_accepts_caller_owned_polygrad_runtime():
    X, y = make_regression(32)
    try:
        import polygrad
    except ModuleNotFoundError as exc:
        if exc.name != "polygrad":
            raise
        return
    pg = polygrad.create(device="cpu")
    try:
        model = SymbolicRegressor({
            **PG_SMALL,
            "population": 12,
            "generations": 1,
            "seed": 20260708,
            "polygrad": pg,
        }).fit(X, y)
        try:
            pred = model.predict(X)
            assert pred.shape == (X.shape[0],)
            assert "polygrad" not in model.get_params()
            loaded = SymbolicRegressor.load(model.save())
            try:
                assert "polygrad" not in loaded.get_params()
                np.testing.assert_allclose(loaded.predict(X), pred, atol=1e-6)
            finally:
                loaded.dispose()
        finally:
            model.dispose()
    finally:
        pg.dispose()


def test_classifier_labels_probabilities_and_load():
    X, y = make_classification()
    model = SymbolicClassifier({**SMALL, "seed": 321}).fit(X, y)
    try:
        assert model.classes == [10.0, 20.0]
        assert model.score(X, y) > 0.85
        pred = model.predict(X)
        assert set(pred.tolist()) <= {10.0, 20.0}
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

        loaded = SymbolicClassifier.load(model.save())
        try:
            assert loaded.classes == [10.0, 20.0]
            assert loaded.score(X, y) > 0.85
        finally:
            loaded.dispose()
    finally:
        model.dispose()


def test_pg_family_classifier_labels_probabilities_and_load():
    X, y = make_classification(36)
    try:
        model = SymbolicClassifier(PG_SMALL).fit(X, y)
    except ModuleNotFoundError as exc:
        if exc.name != "polygrad":
            raise
        return
    try:
        assert model.classes == [10.0, 20.0]
        pred = model.predict(X)
        assert pred.shape == (X.shape[0],)
        assert set(pred.tolist()) <= {10.0, 20.0}
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

        loaded = SymbolicClassifier.load(model.save())
        try:
            assert loaded.classes == [10.0, 20.0]
            np.testing.assert_allclose(loaded.predict_proba(X), proba, atol=1e-6)
        finally:
            loaded.dispose()
    finally:
        model.dispose()


def test_modular_search_options_persist():
    X, y = make_regression(96)
    params = {
        **SMALL,
        "population": 96,
        "generations": 30,
        "maxNodes": 21,
        "maxDepth": 5,
        "frontierSize": 10,
        "validationFraction": 0,
        "islands": 3,
        "migrationInterval": 5,
        "migrationCount": 1,
        "warmupGenerations": 12,
        "warmupMinNodes": 7,
        "broodSize": 3,
        "rowSampleSize": 48,
        "localRefineInterval": 6,
        "localRefineCount": 3,
        "complexityHofSize": 24,
        "finalSelector": "loss",
        "complexityBucketWidth": 2.0,
        "seed": 778,
    }
    model = SymbolicRegressor(params).fit(X, y)
    try:
        assert model.score(X, y) > 0.75
        loaded = SymbolicRegressor.load(model.save())
        try:
            loaded_params = loaded.get_params()
            assert loaded_params["islands"] == 3
            assert loaded_params["broodSize"] == 3
            assert loaded_params["complexityHofSize"] == 24
            assert loaded_params["finalSelector"] == "loss"
            assert loaded.score(X, y) > 0.75
        finally:
            loaded.dispose()
    finally:
        model.dispose()


def test_default_search_spaces_expose_advanced_search_controls():
    common_expected = {
        "population": {"type": "int_uniform", "low": 128, "high": 1024},
        "generations": {"type": "int_uniform", "low": 80, "high": 400},
        "maxNodes": {"type": "int_uniform", "low": 7, "high": 63, "condition": {"strategy": "tree"}},
        "operatorSet": {"type": "categorical", "values": ["full", "smooth", "basic"]},
        "complexityPenalty": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
        "islands": [1, 2, 4],
        "broodSize": [1, 2],
        "complexityHofSize": [0, 64, 128],
        "finalSelector": ["objective", "loss"],
    }
    for cls in (SymbolicRegressor, SymbolicClassifier):
        space = cls.default_search_space()
        for key, spec in common_expected.items():
            if isinstance(spec, list):
                assert space[key]["type"] == "categorical"
                assert space[key]["values"] == spec
            else:
                assert space[key] == spec
        for key in ("maxNodes", "broodSize", "complexityHofSize", "finalSelector"):
            assert space[key]["condition"] == {"strategy": "tree"}
        for key in ("terms", "ridge", "immigrantRate"):
            assert space[key]["condition"] == {"strategy": "family"}
        assert space["strategy"]["values"] == ["tree", "family"]
        assert space["backend"]["values"] == ["c"]
        assert space["mutationRate"] == {"type": "uniform", "low": 0.15, "high": 0.55}
        assert space["crossoverRate"] == {"type": "uniform", "low": 0.35, "high": 0.75}

    transformer_space = FormulaTransformer.default_search_space()
    for key in ["topK", *common_expected.keys()]:
        assert key in transformer_space
    for key, values in {
        "islands": [1, 2, 4],
        "broodSize": [1, 2],
        "complexityHofSize": [0, 64, 128],
        "finalSelector": ["objective", "loss"],
    }.items():
        assert transformer_space[key]["values"] == values
    assert "mutationRate" not in transformer_space
    assert "crossoverRate" not in transformer_space
    assert "loss" in SymbolicRegressor.default_search_space()
    assert "loss" not in SymbolicClassifier.default_search_space()
    assert "loss" not in transformer_space


def test_regressor_polygrad_refinement_commits_improving_constants():
    X, y = make_regression(64)
    model = SymbolicRegressor({**SMALL, "seed": 123}).fit(X, y)
    try:
        formula = model.formula()
        const_index = next((i for i, node in enumerate(formula["nodes"]) if node["op"] == "const"), -1)
        assert const_index >= 0
        model._set_formula_constant(0, const_index, formula["nodes"][const_index]["value"] + 1)
        damaged_score = model.score(X, y)
        try:
            report = model.refine_polygrad(X, y, epochs=40, lr=0.01)
        except ModuleNotFoundError as exc:
            if exc.name != "polygrad":
                raise
            return
        assert report["committed"]
        assert report["afterLoss"] <= report["beforeLoss"] + 1e-10
        assert model.score(X, y) >= damaged_score
        loaded = SymbolicRegressor.load(model.save())
        try:
            assert loaded.score(X, y) >= damaged_score
        finally:
            loaded.dispose()
    finally:
        model.dispose()


def test_polygrad_refinement_supports_protected_division_formulas():
    X = np.asarray([[0], [0.5], [1], [2], [4], [8]], dtype=np.float64)
    y = 2 / (X[:, 0] + 1)
    formula = {
        "nodes": [
            {"op": "const", "value": 1.5},
            {"op": "var", "feature": 0},
            {"op": "const", "value": 0.6},
            {"op": "add", "left": 1, "right": 2},
            {"op": "div", "left": 0, "right": 3},
        ]
    }
    try:
        report = refine_formula_polygrad(formula, X, y, epochs=2, lr=0.001)
    except ModuleNotFoundError as exc:
        if exc.name != "polygrad":
            raise
        return
    assert report["status"] == "ok", report.get("reason", report["status"])
    assert np.isfinite(report["loss"])


def test_transformer_outputs_top_k_features():
    X, y = make_regression(64)
    model = FormulaTransformer({**SMALL, "topK": 4, "seed": 555}).fit(X, y)
    try:
        Z = model.transform(X)
        assert Z.shape == (X.shape[0], 4)
        assert np.isfinite(Z).all()
        loaded = FormulaTransformer.load(model.save())
        try:
            assert loaded.transform(X).shape == Z.shape
        finally:
            loaded.dispose()
    finally:
        model.dispose()


def test_verifier_affine_monotonicity():
    formula = {
        "nodes": [
            {"op": "var", "opId": 1, "left": -1, "right": -1, "feature": 0, "value": 0},
            {"op": "const", "opId": 0, "left": -1, "right": -1, "feature": -1, "value": 2},
            {"op": "mul", "opId": 4, "left": 0, "right": 1, "feature": -1, "value": 0},
        ]
    }
    report = FormulaVerifier.verify(
        formula,
        {"n_features": 1, "checks": ["domain", {"kind": "monotonicity", "feature": 0, "direction": "increasing"}]},
    )
    assert report["status"] == "proved"


def test_polygrad_protected_nodes_and_borrowed_runtime():
    try:
        import polygrad
    except ModuleNotFoundError as exc:
        if exc.name != 'polygrad':
            raise
        import pytest
        pytest.skip('optional polygrad is not installed')
    from wlearn_sym._polygrad import evaluate_formula_polygrad
    pg = polygrad.create(device="cpu")
    formula = {"nodes": [{"op": "const", "value": 40}, {"op": "exp", "left": 0}]}
    try:
        actual = evaluate_formula_polygrad(formula, [[0], [1]], polygrad=pg)
        np.testing.assert_allclose(actual, [1e12, 1e12], rtol=1e-7)
        # Caller runtime remains usable after temporary model disposal.
        t = pg.Tensor([3.0])
        try:
            np.testing.assert_array_equal(t.numpy(), [3.0])
        finally:
            t.dispose()
    finally:
        pg.dispose()

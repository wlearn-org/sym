"""Benchmark integrity: independent splits and an independent formula evaluator."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_data import dataset, manifest
from quality_reference import predict
from wlearn_sym import SymbolicRegressor


def test_manifest_is_reproducible_and_test_size_cannot_change_training():
    tasks = manifest()["tasks"]
    assert len({t["id"] for t in tasks}) == len(tasks) == 24
    for task in tasks:
        first = dataset(task, 48, 64)
        same = dataset(task, 48, 64)
        different_test_size = dataset(task, 48, 99)
        for a, b in zip(first, same):
            np.testing.assert_array_equal(a, b)
            assert np.isfinite(a).all()
        for i in (0, 1):
            np.testing.assert_array_equal(first[i], different_test_size[i])


@pytest.mark.parametrize(
    "strategy,hierarchical", [("tree", False), ("family", False), ("family", True)]
)
def test_exported_formula_reference(strategy, hierarchical):
    rng = np.random.default_rng(412)
    X = rng.uniform(-2, 2, (64, 3))
    y = np.sin(X[:, 0] * 2.3) + X[:, 1] * X[:, 2]
    options = dict(strategy=strategy, population=16, eliteCount=4, generations=4)
    if strategy == "family":
        options.update(hierarchical=hierarchical, terms=4)
    model = SymbolicRegressor(options)
    try:
        model.fit(X, y)
        Xt = rng.uniform(-3, 3, (100, 3))
        np.testing.assert_allclose(
            model.predict(Xt), predict(model.formula(), Xt), atol=1e-5, rtol=1e-5
        )
    finally:
        model.dispose()


def test_reference_keeps_protected_division_and_hierarchy():
    formula = dict(
        bias=0,
        terms=[
            dict(op="div", featureA=0, featureB=1, p0=1, p1=0, coefficient=0),
            dict(op="sin", featureA=2, featureB=0, p0=1, p1=0, coefficient=1),
        ],
    )
    X = np.array([[1e-7, 0], [1e-7, -1e-9]])
    expected = np.sin(
        np.array([1e-7, -1e-7], dtype=np.float32).astype(float) / 1e-6
    ).astype(np.float32)
    np.testing.assert_array_equal(predict(formula, X), expected)


def test_scale_manifest_pairs_units_without_changing_observations():
    m = manifest(Path(__file__).with_name("quality-scale-tasks.json"))
    groups = {}
    for task in m["tasks"]:
        X, y, Xt, yt = dataset(task, 48, 64, data_seed=m["data_seed"])
        original = groups.setdefault(
            task["data_id"],
            (X, y / task["target_scale"], Xt, yt / task["target_scale"]),
        )
        for a, b in zip(
            (X, y / task["target_scale"], Xt, yt / task["target_scale"]), original
        ):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        larger_test = dataset(task, 48, 99, data_seed=m["data_seed"])
        np.testing.assert_array_equal(X, larger_test[0])
        np.testing.assert_array_equal(y, larger_test[1])
    assert len(groups) == 8

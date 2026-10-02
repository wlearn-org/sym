import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wlearn_sym import SymbolicRegressor


JS_TRAIN = r"""
const fs = require('fs')
const { SymbolicRegressor } = require('./js/src');
function makeRegression(n) {
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const x0 = -2 + 4 * i / (n - 1)
    const x1 = Math.sin(i * 0.37)
    X.push([x0, x1])
    y.push(2 * x0 - 0.5 * x1 + 1)
  }
  return { X, y }
}
(async () => {
  const out = process.argv[1]
  const { X, y } = makeRegression(64)
  const model = await SymbolicRegressor.create({
    population: 64, generations: 35, maxNodes: 15, maxDepth: 4,
    frontierSize: 6, eliteCount: 4, tournamentSize: 4,
    validationFraction: 0.2, complexityPenalty: 0.0001,
    operatorSet: 'basic', seed: 2026
  })
  model.fit(X, y)
  fs.writeFileSync(out + '/js.wlrn', model.save())
  fs.writeFileSync(out + '/js.json', JSON.stringify({ X, y, pred: Array.from(model.predict(X)) }))
  model.dispose()
})().catch(err => { console.error(err); process.exit(1) })
"""


JS_LOAD = r"""
const fs = require('fs')
const { SymbolicRegressor } = require('./js/src');
(async () => {
  const dir = process.argv[1]
  const side = JSON.parse(fs.readFileSync(dir + '/js.json', 'utf8'))
  const model = await SymbolicRegressor.load(fs.readFileSync(dir + '/py.wlrn'))
  const pred = Array.from(model.predict(side.X))
  fs.writeFileSync(dir + '/js_from_py.json', JSON.stringify({ pred }))
  model.dispose()
})().catch(err => { console.error(err); process.exit(1) })
"""


JS_TRAIN_PG = r"""
const fs = require('fs')
const { SymbolicRegressor } = require('./js/src');
function makeRegression(n) {
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const x0 = -2 + 4 * i / (n - 1)
    const x1 = Math.sin(i * 0.37)
    X.push([x0, x1])
    y.push(2 * x0 - 0.5 * x1 + 1)
  }
  return { X, y }
}
(async () => {
  const out = process.argv[1]
  const { X, y } = makeRegression(32)
  const model = await SymbolicRegressor.create({
    engine: 'pg-family',
    population: 10,
    generations: 1,
    terms: 3,
    eliteCount: 2,
    frontierSize: 3,
    seed: 20260705,
    polygrad: { core: 'wasm', device: 'cpu' }
  })
  await model.fit(X, y)
  const pred = Array.from(await model.predict(X))
  fs.writeFileSync(out + '/js-pg.wlrn', model.save())
  fs.writeFileSync(out + '/js-pg.json', JSON.stringify({ X, y, pred }))
  model.dispose()
})().catch(err => { console.error(err); process.exit(1) })
"""


JS_LOAD_PG = r"""
const fs = require('fs')
const { SymbolicRegressor } = require('./js/src');
(async () => {
  const dir = process.argv[1]
  const side = JSON.parse(fs.readFileSync(dir + '/py-pg.json', 'utf8'))
  const model = await SymbolicRegressor.load(fs.readFileSync(dir + '/py-pg.wlrn'))
  const pred = Array.from(await model.predict(side.X))
  fs.writeFileSync(dir + '/js-from-py-pg.json', JSON.stringify({ pred }))
  model.dispose()
})().catch(err => { console.error(err); process.exit(1) })
"""


def _run_node(code, tmp):
    env = os.environ.copy()
    local_pg = Path(__file__).parents[3] / "polygrad" / "polygrad" / "js"
    if local_pg.exists():
        env.setdefault("WLEARN_SYM_POLYGRAD_JS", str(local_pg))
    subprocess.run(
        ["node", "-e", code, str(tmp)],
        cwd=Path(__file__).parents[1],
        env=env,
        check=True,
    )


def test_js_python_js_bundle_parity():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _run_node(JS_TRAIN, tmp)
        side = json.loads((tmp / "js.json").read_text(encoding="utf-8"))
        X = np.asarray(side["X"], dtype=np.float64)
        js_pred = np.asarray(side["pred"], dtype=np.float64)

        py_model = SymbolicRegressor.load(tmp / "js.wlrn")
        try:
            np.testing.assert_allclose(py_model.predict(X), js_pred, atol=1e-12)
            py_model.save(tmp / "py.wlrn")
        finally:
            py_model.dispose()

        _run_node(JS_LOAD, tmp)
        js_from_py = np.asarray(
            json.loads((tmp / "js_from_py.json").read_text(encoding="utf-8"))["pred"],
            dtype=np.float64,
        )
        np.testing.assert_allclose(js_from_py, js_pred, atol=1e-12)


def test_pg_family_js_python_bundle_parity():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _run_node(JS_TRAIN_PG, tmp)
        side = json.loads((tmp / "js-pg.json").read_text(encoding="utf-8"))
        X = np.asarray(side["X"], dtype=np.float64)
        js_pred = np.asarray(side["pred"], dtype=np.float64)

        py_loaded = SymbolicRegressor.load(tmp / "js-pg.wlrn")
        try:
            np.testing.assert_allclose(py_loaded.predict(X), js_pred, atol=1e-5)
        finally:
            py_loaded.dispose()

        try:
            py_model = SymbolicRegressor(
                {
                    "engine": "pg-family",
                    "population": 10,
                    "generations": 1,
                    "terms": 3,
                    "eliteCount": 2,
                    "frontierSize": 3,
                    "seed": 20260706,
                }
            ).fit(X, np.asarray(side["y"], dtype=np.float64))
        except ModuleNotFoundError as exc:
            if exc.name == "polygrad":
                pytest.skip("Python Polygrad unavailable")
            raise
        try:
            py_pred = py_model.predict(X)
            py_model.save(tmp / "py-pg.wlrn")
            (tmp / "py-pg.json").write_text(
                json.dumps({"X": side["X"], "pred": py_pred.tolist()}), encoding="utf-8"
            )
        finally:
            py_model.dispose()

        _run_node(JS_LOAD_PG, tmp)
        js_from_py = np.asarray(
            json.loads((tmp / "js-from-py-pg.json").read_text(encoding="utf-8"))[
                "pred"
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(js_from_py, py_pred, atol=1e-5)


if __name__ == "__main__":
    test_js_python_js_bundle_parity()
    print("sym cross-language parity passed")


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("classes", [0, 2, 3])
@pytest.mark.parametrize("method", ["coordinate", "lm"])
def test_c_family_native_wasm_and_bundle_parity(classes, hierarchical, method):
    from wlearn_sym import SymbolicClassifier

    params = dict(
        strategy="family",
        backend="c",
        population=24,
        generations=4,
        terms=4,
        eliteCount=4,
        islands=3,
        validationFraction=0.2,
        operatorSet="full",
        seed=194,
        hierarchical=hierarchical,
        polishPasses=2,
        localRefineInterval=1,
        localRefineCount=2,
        polishBatchSize=16 if method == "coordinate" else 0,
        polishMethod=method,
        scaleAware=method == "lm",
    )
    X = np.random.default_rng(420).uniform(-2, 2, (60, 3))
    labels = [91, -7, 123][:classes]
    y = (
        np.asarray(labels)[np.arange(len(X)) % classes]
        if classes
        else np.sin(X[:, 0]) + X[:, 1] * X[:, 2]
    )
    if classes:
        params["classes"] = labels
    Model = SymbolicClassifier if classes else SymbolicRegressor
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "input.json").write_text(
            json.dumps(dict(X=X.tolist(), y=y.tolist(), params=params, classes=classes))
        )
        _run_node(
            r"""
const fs = require('node:fs')
const { SymbolicRegressor, SymbolicClassifier } = require('./js/src')
;(async () => {
  const dir = process.argv[1], input = JSON.parse(fs.readFileSync(dir+'/input.json'))
  const Model = input.classes ? SymbolicClassifier : SymbolicRegressor
  const m = await Model.create(input.params)
  try {
    m.fit(input.X, input.y)
    fs.writeFileSync(dir+'/js.wlrn', m.save())
    fs.writeFileSync(dir+'/js.json', JSON.stringify({pred: Array.from(m.predict(input.X)),
      scores: Array.from(input.classes ? m.predictProba(input.X) : m.predict(input.X))}))
  } finally { m.dispose() }
})().catch(e => { console.error(e); process.exitCode=1 })
""",
            tmp,
        )
        side = json.loads((tmp / "js.json").read_text())
        native = Model(params).fit(X, y)
        loaded = Model.load(tmp / "js.wlrn")
        try:
            for m in (native, loaded):
                np.testing.assert_allclose(
                    m.predict(X), side["pred"], rtol=1e-6, atol=1e-6
                )
                scores = m.predict_proba(X) if classes else m.predict(X)
                np.testing.assert_allclose(
                    scores.reshape(-1), side["scores"], rtol=1e-6, atol=1e-6
                )
            native.save(tmp / "native.wlrn")
            loaded.save(tmp / "roundtrip.wlrn")
            from wlearn.bundle import decode_bundle

            original = decode_bundle(tmp / "js.wlrn")
            roundtrip = decode_bundle(tmp / "roundtrip.wlrn")
            assert original[0] == roundtrip[0]
            assert original[2] == roundtrip[2]
        finally:
            native.dispose()
            loaded.dispose()
        _run_node(
            r"""
const fs = require('node:fs'), assert = require('node:assert/strict')
const { SymbolicRegressor, SymbolicClassifier } = require('./js/src')
;(async () => {
  const dir=process.argv[1], input=JSON.parse(fs.readFileSync(dir+'/input.json'))
  const side=JSON.parse(fs.readFileSync(dir+'/js.json'))
  const Model=input.classes ? SymbolicClassifier : SymbolicRegressor
  for (const name of ['native', 'roundtrip']) {
    const m=await Model.load(fs.readFileSync(dir+'/'+name+'.wlrn'))
    try {
      const scores=input.classes ? m.predictProba(input.X) : m.predict(input.X)
      scores.forEach((v,i) => assert.ok(Math.abs(v-side.scores[i]) <= 1e-6*(1+Math.abs(v))))
    } finally { m.dispose() }
  }
})().catch(e => { console.error(e); process.exitCode=1 })
""",
            tmp,
        )

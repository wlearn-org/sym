'use strict'
const assert = require('node:assert/strict')
const { SymbolicRegressor, SymbolicClassifier, FormulaTransformer } = require('../js/src')
const { createRequire } = require('node:module')
const { ValidationError, Pipeline, load, decodeBundle, encodeBundle } = createRequire(require('node:path').resolve(__dirname, '../js/package.json'))('@wlearn/core')
const { CFamilyEngine } = require('../js/src/family-c')
const { PgFamilyRegressorEngine } = require('../js/src/engine-pg-family')
const p = { strategy: 'family', backend: 'c', population: 24, generations: 4, terms: 3, eliteCount: 4, frontierSize: 4, islands: 3, validationFraction: .2, seed: 123, operatorSet: 'basic', polishPasses: 2 }
async function main() {
  const X = Array.from({ length: 60 }, (_, i) => [(i - 30) / 10, Math.sin(i * .47)])
  const y = X.map(([a, b]) => 2 * a + .3 * b + 1)
  for (const params of [{ strategy: 'bogus' }, { backend: 'bogus' }, { strategy: 'family', backend: 'polygrad' }, { strategy: 'tree', backend: 'polygrad' }, { engine: 'c', strategy: 'family' }]) {
    await assert.rejects(SymbolicRegressor.create(params), ValidationError)
  }
  await assert.rejects(FormulaTransformer.create(p), ValidationError)
  const m = await SymbolicRegressor.create(p)
  let restored, pipeline
  try {
    assert.equal(m.fit({ rows: X.length, cols: 2, data: Float64Array.from(X.flat()) }, y), m)
    assert.ok(m.score(X, y) > .98)
    const pred = m.predict(X)
    await assert.rejects(m.predictPolygrad(X), /not implemented/)
    assert.ok(!(pred instanceof Promise))
    const decoded = decodeBundle(m.save())
    assert.equal(decoded.manifest.typeId, 'wlearn.sym.regressor@2')
    const entry = decoded.toc.find(e => e.id === 'model')
    const mismatched = encodeBundle({ ...decoded.manifest, typeId: SymbolicClassifier.typeId.replace('@1', '@2') },
      [{ id: 'model', mediaType: entry.mediaType, data: decoded.blobs.slice(entry.offset, entry.offset + entry.length) }])
    await assert.rejects(load(mismatched), /payload task/)
    restored = await load(m.save())
    assert.deepEqual(restored.predict(X), pred)
    assert.equal(restored.getParams().strategy, 'family')
    assert.equal(restored.capabilities.activeBackend, 'c')
    assert.throws(() => m.setParams({ backend: 'polygrad' }), ValidationError)
    assert.deepEqual(m.predict(X), pred)
    m.setParams({ terms: 0 })
    assert.throws(() => m.fit(X, y), ValidationError)
    assert.deepEqual(m.predict(X), pred)
    m.setParams({ terms: 3 })
    const state = m._familyEngine.toState()
    const old = PgFamilyRegressorEngine.fromStateSync(state)
    assert.deepEqual(Array.from(await old.predict(X)), Array.from(pred))
    old.dispose()
    const again = CFamilyEngine.fromState(state)
    assert.deepEqual(again.predict(X), pred)
    again.dispose()
    for (const patch of [{ nFeatures: 2 ** 32 + 2 }, { fitted: false }, { best: null }]) {
      assert.throws(() => CFamilyEngine.fromState({ ...state, ...patch }), ValidationError)
    }
    for (const option of ['hierarchical', 'migrationInterval', 'localRefineCount', 'scoreMode', 'jit']) {
      m.setParams({ [option]: 1 })
      assert.throws(() => m.fit(X, y), ValidationError, option)
      m.setParams({ [option]: null })
    }
    const bad = structuredClone(state)
    bad.best.fa[0] = -1
    assert.throws(() => CFamilyEngine.fromState(bad), ValidationError)
    pipeline = new Pipeline([['family', restored]])
    assert.equal(pipeline.fit(X, y), pipeline)
    const loaded = await load(pipeline.save())
    try { assert.deepEqual(loaded.predict(X), pipeline.predict(X)) } finally { loaded.dispose() }
    console.log('C family: sync fit, dense matrices, persistence, legacy semantics, refit failure and Pipeline passed')
  } finally { pipeline?.dispose(); restored?.dispose(); m.dispose() }
  for (const count of [2, 3]) {
    const labels = X.map((_, i) => [91, -7, 123][i % count])
    const c = await SymbolicClassifier.create({ ...p, classes: [91, -7, 123].slice(0, count) })
    try {
      assert.equal(c.fit(X, labels), c)
      assert.deepEqual(c.classes, [91, -7, 123].slice(0, count))
      const probs = c.predictProba(X)
      for (let i = 0; i < X.length; i++) assert.ok(Math.abs(probs.slice(i * count, (i + 1) * count).reduce((a, b) => a + b, 0) - 1) < 1e-12)
      const restored = await load(c.save())
      try { assert.deepEqual(restored.predictProba(X), probs); assert.deepEqual(restored.predict(X), c.predict(X)) } finally { restored.dispose() }
      console.log(`C family classifier: ${count} classes passed`)
    } finally { c.dispose() }
  }
}
main().catch(e => { console.error(e); process.exitCode = 1 })

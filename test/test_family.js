'use strict'
const assert = require('node:assert/strict')
const { SymbolicRegressor, SymbolicClassifier, FormulaTransformer } = require('../js/src')
const { createRequire } = require('node:module')
const { ValidationError, Pipeline, load, decodeBundle, encodeBundle } = createRequire(
  require('node:path').resolve(__dirname, '../js/package.json')
)('@wlearn/core')
const { CFamilyEngine } = require('../js/src/family-c')
const legacy = require('./fixtures/legacy-family.json')
const p = {
  strategy: 'family',
  backend: 'c',
  population: 24,
  generations: 4,
  terms: 3,
  eliteCount: 4,
  frontierSize: 4,
  islands: 3,
  validationFraction: 0.2,
  seed: 123,
  operatorSet: 'basic',
  polishPasses: 2
}
async function main() {
  const X = Array.from({ length: 60 }, (_, i) => [(i - 30) / 10, Math.sin(i * 0.47)])
  const y = X.map(([a, b]) => 2 * a + 0.3 * b + 1)
  for (const params of [
    { strategy: 'bogus' },
    { backend: 'bogus' },
    { strategy: 'tree', backend: 'polygrad' },
    { engine: 'c', strategy: 'family' }
  ]) {
    await assert.rejects(SymbolicRegressor.create(params), ValidationError)
  }
  await assert.rejects(FormulaTransformer.create(p), ValidationError)
  const m = await SymbolicRegressor.create(p)
  let restored, pipeline
  try {
    assert.equal(m.fit({ rows: X.length, cols: 2, data: Float64Array.from(X.flat()) }, y), m)
    assert.ok(m.score(X, y) > 0.98)
    for (const index of [0.5, -1, 2 ** 32, NaN, true]) {
      await assert.rejects(m.refinePolygrad(X, y, { index, epochs: 1 }), /invalid family head index/)
    }
    const pred = m.predict(X)
    assert.equal(m.capabilities.polygradExecution, true)
    assert.ok(!(pred instanceof Promise))
    const decoded = decodeBundle(m.save())
    assert.equal(decoded.manifest.typeId, 'wlearn.sym.regressor@2')
    const entry = decoded.toc.find(e => e.id === 'model')
    const mismatched = encodeBundle(
      { ...decoded.manifest, typeId: SymbolicClassifier.typeId.replace('@1', '@2') },
      [
        {
          id: 'model',
          mediaType: entry.mediaType,
          data: decoded.blobs.slice(entry.offset, entry.offset + entry.length)
        }
      ]
    )
    await assert.rejects(load(mismatched), /payload task/)
    restored = await load(m.save())
    assert.deepEqual(restored.predict(X), pred)
    assert.equal(restored.getParams().strategy, 'family')
    assert.equal(restored.capabilities.activeBackend, 'c')
    assert.throws(() => m.setParams({ backend: 'bogus' }), ValidationError)
    assert.deepEqual(m.predict(X), pred)
    m.setParams({ terms: 0 })
    assert.throws(() => m.fit(X, y), ValidationError)
    assert.deepEqual(m.predict(X), pred)
    m.setParams({ terms: 3 })
    const state = m._familyEngine.toState()
    const again = CFamilyEngine.fromState(state)
    assert.deepEqual(again.predict(X), pred)
    again.dispose()
    for (const patch of [{ nFeatures: 2 ** 32 + 2 }, { fitted: false }, { best: null }]) {
      assert.throws(() => CFamilyEngine.fromState({ ...state, ...patch }), ValidationError)
    }
    for (const option of [
      'hierarchical',
      'migrationInterval',
      'localRefineCount',
      'scoreMode',
      'jit',
      'stackSummaries',
      'evalCacheSize'
    ]) {
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
    try {
      assert.deepEqual(loaded.predict(X), pipeline.predict(X))
    } finally {
      loaded.dispose()
    }
    console.log(
      'C family: sync fit, dense matrices, persistence, legacy semantics, refit failure and Pipeline passed'
    )
  } finally {
    pipeline?.dispose()
    restored?.dispose()
    m.dispose()
  }
  const chain = {
    terms: 32,
    fa: [0, ...Array.from({ length: 31 }, (_, i) => 1 + i)],
    fb: [0, ...Array.from({ length: 31 }, (_, i) => 1 + i)],
    op: Array(32).fill(0),
    p0: Array(32).fill(1),
    p1: Array(32).fill(0),
    coef: Array(32).fill(1),
    bias: 0
  }
  const text = require('../js/src/family-format').formulaText(chain, ['x0'])
  assert(text.includes('t31 = (t30+t30)') && text.length < 5000)
  for (const fixture of legacy.bundles) {
    const Model = fixture.classes ? SymbolicClassifier : SymbolicRegressor
    const m = await Model.load(require('node:path').join(__dirname, 'fixtures', fixture.file))
    let restored
    try {
      const prediction = m.predict(fixture.X)
      for (let i = 0; i < prediction.length; i++)
        assert(Math.abs(prediction[i] - fixture.prediction[i]) < 1e-6)
      if (fixture.classes) {
        const p = m.predictProba(fixture.X),
          expected = fixture.probability.flat()
        for (let i = 0; i < p.length; i++) assert(Math.abs(p[i] - expected[i]) < 1e-6)
      }
      restored = await Model.load(m.save())
      assert.deepEqual(restored.predict(fixture.X), prediction)
    } finally {
      restored?.dispose()
      m.dispose()
    }
  }
  for (const fixture of legacy.operators) {
    const m = CFamilyEngine.fromState(fixture.state)
    try {
      const actual = m.predict(legacy.operatorX)
      for (let i = 0; i < actual.length; i++)
        assert(
          Math.abs(actual[i] - fixture.prediction[i]) <= 1e-7 * Math.max(1, Math.abs(actual[i]))
        )
    } finally {
      m.dispose()
    }
  }
  for (const count of [2, 3]) {
    const labels = X.map((_, i) => [91, -7, 123][i % count])
    const c = await SymbolicClassifier.create({ ...p, classes: [91, -7, 123].slice(0, count) })
    try {
      assert.equal(c.fit(X, labels), c)
      assert.deepEqual(c.classes, [91, -7, 123].slice(0, count))
      const probs = c.predictProba(X)
      for (let i = 0; i < X.length; i++)
        assert.ok(
          Math.abs(probs.slice(i * count, (i + 1) * count).reduce((a, b) => a + b, 0) - 1) < 1e-12
        )
      const restored = await load(c.save())
      try {
        assert.deepEqual(restored.predictProba(X), probs)
        assert.deepEqual(restored.predict(X), c.predict(X))
      } finally {
        restored.dispose()
      }
      console.log(`C family classifier: ${count} classes passed`)
    } finally {
      c.dispose()
    }
  }
}
main().catch(e => {
  console.error(e)
  process.exitCode = 1
})

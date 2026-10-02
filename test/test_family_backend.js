'use strict'
const assert = require('node:assert/strict')
const { SymbolicRegressor, SymbolicClassifier } = require('../js/src')
const { createRequire } = require('node:module')
const pkgRequire = createRequire(require('node:path').resolve(__dirname, '../js/package.json'))
async function main() {
  const { FamilyScorer } = require('../js/src/family-scorer')
  const createScorer = FamilyScorer.create
  const batches = []
  FamilyScorer.create = function (...args) {
    batches.push(args[5])
    return createScorer.apply(this, args)
  }
  const defaultBatch = await SymbolicRegressor.create({
    strategy: 'family',
    backend: 'polygrad',
    population: 8,
    eliteCount: 2,
    generations: 1,
    terms: 2
  })
  try {
    await defaultBatch.fit([[0], [1], [2], [3]], [0, 1, 2, 3])
    assert.deepEqual(batches, [8])
  } finally {
    defaultBatch.dispose()
    FamilyScorer.create = createScorer
  }

  const previous = process.env.POLY_DEV
  const envModel = await SymbolicRegressor.create({
    strategy: 'family',
    backend: 'polygrad',
    population: 8,
    eliteCount: 2,
    generations: 1,
    terms: 2,
    batchSize: 8
  })
  try {
    process.env.POLY_DEV = 'invalid-sym-test-device'
    await assert.rejects(envModel.fit([[0], [1], [2], [3]], [0, 1, 2, 3]), /device|target/i)
  } finally {
    if (previous === undefined) delete process.env.POLY_DEV
    else process.env.POLY_DEV = previous
    envModel.dispose()
  }
  for (const polishMethod of ['coordinate', 'lm'])
  for (const classes of [0, 2, 3])
    for (const hierarchical of [false, true]) {
      const X = Array.from({ length: 64 }, (_, i) => [
        (i - 32) / 32,
        Math.sin(i * 0.47),
        Math.cos(i * 0.31)
      ])
      const y = X.map((row, i) => (classes ? i % classes : Math.sin(row[0]) + row[1] * row[2]))
      const Model = classes ? SymbolicClassifier : SymbolicRegressor
      const params = {
        strategy: 'family',
        population: 16,
        eliteCount: 4,
        generations: 2,
        terms: 3,
        polishPasses: 2,
        polishBatchSize: polishMethod === 'lm' ? 0 : 20,
        polishMethod,
        scaleAware: polishMethod === 'lm',
        seed: 11,
        validationFraction: 0.2,
        localRefineInterval: 1,
        localRefineCount: 2,
        hierarchical
      }
      const c = await Model.create(params)
      const pg = await Model.create({
        ...params,
        backend: 'polygrad',
        batchSize: 16,
        polygrad: { core: 'native' }
      })
      try {
        assert.equal(c.fit(X, y), c)
        const fit = pg.fit(X, y)
        assert(fit instanceof Promise)
        await fit
        const a = c.predict(X),
          b = pg.predict(X)
        for (let i = 0; i < a.length; i++)
          assert(Math.abs(a[i] - b[i]) < 1e-5, `prediction ${classes}/${i}: ${a[i]} vs ${b[i]}`)
        const devicePred = await pg.predictPolygrad(X)
        for (let i = 0; i < b.length; i++) assert(Math.abs(devicePred[i] - b[i]) < 1e-5)
        for (let index = 0; index < (classes > 2 ? classes : 1); index++) {
          const report = await pg.refinePolygrad(X, y, { index, epochs: 2 })
          assert.equal(report.status, 'ok')
          assert(!report.committed || report.afterObjective <= report.beforeObjective + 1e-10)
        }
        const bytes = pg.save()
        assert.equal(String.fromCharCode(...bytes.subarray(0, 4)), 'WLRN')
        const loaded = await Model.load(bytes)
        try {
          assert.deepEqual(loaded.predict(X), pg.predict(X))
        } finally {
          loaded.dispose()
        }
      } finally {
        c.dispose()
        pg.dispose()
      }
    }
  const X = Array.from({ length: 64 }, (_, i) => [(i - 32) / 32, Math.sin(i)])
  const y = X.map(row => Math.sin(2.3 * row[0] + 0.4) + 0.1 * row[1])
  const m = await SymbolicRegressor.create({
    strategy: 'family',
    population: 16,
    generations: 2,
    terms: 3,
    seed: 8,
    validationFraction: 0.2
  })
  try {
    m.fit(X, y)
    const report = await m.refinePolygrad(X, y, {
      epochs: 8,
      lr: 0.01,
      polygrad: { core: 'native', device: 'cpu' }
    })
    assert.equal(report.status, 'ok')
    assert.equal(report.history.length, 8)
    if (report.committed) assert(report.afterObjective <= report.beforeObjective + 1e-10)
    const loaded = await SymbolicRegressor.load(m.save())
    try {
      assert.deepEqual(loaded.predict(X), m.predict(X))
    } finally {
      loaded.dispose()
    }
  } finally {
    m.dispose()
  }
  const pgModule = process.env.WLEARN_SYM_POLYGRAD_JS
    ? require(process.env.WLEARN_SYM_POLYGRAD_JS)
    : pkgRequire('polygrad')
  const runtime = await pgModule.create({
    core: 'native'
  })
  try {
    for (let round = 0; round < 2; round++) {
      const model = await SymbolicRegressor.create({
        strategy: 'family',
        backend: 'polygrad',
        polygrad: runtime,
        population: 8,
        eliteCount: 2,
        generations: 2,
        terms: 2,
        batchSize: 8,
        operatorSet: 'basic'
      })
      try {
        await model.fit(X, y)
        const before = model.predict(X)
        model.setParams({ scorerDtype: 'bad' })
        await assert.rejects(model.fit(X, y), /scorerDtype/)
        assert.deepEqual(model.predict(X), before)
        await assert.rejects(model.predictPolygrad({ rows: 2, cols: 2, data: [1] }), /data.length/)
      } finally {
        model.dispose()
      }
      assert.equal(runtime.stats().jit.liveCount, 0)
      const probe = runtime.Tensor.full([1], 3)
      try {
        assert.deepEqual(Array.from(await probe.toTypedArrayAsync()), [3])
      } finally {
        probe.dispose()
      }
    }
    let release
    const supplied = new Promise(resolve => {
      release = resolve
    })
    const cancelled = await SymbolicRegressor.create({
      strategy: 'family',
      backend: 'polygrad',
      polygrad: supplied,
      population: 8,
      eliteCount: 2,
      generations: 1,
      terms: 2,
      batchSize: 8
    })
    const pending = cancelled.fit(X, y)
    assert.throws(() => cancelled.setParams({ seed: 17 }), /during fit/)
    assert.throws(() => cancelled.fit(X, y), /in progress/)
    cancelled.dispose()
    release(runtime)
    await assert.rejects(pending, /disposed during fit/)
    assert.equal(runtime.stats().jit.liveCount, 0)
  } finally {
    runtime.dispose()
  }
  const { Pipeline, load } = pkgRequire('@wlearn/core')
  const pipeline = new Pipeline([
    [
      'family',
      await SymbolicRegressor.create({
        strategy: 'family',
        backend: 'polygrad',
        terms: 2,
        batchSize: 8,
        population: 8,
        eliteCount: 2,
        generations: 1,
        polygrad: { core: 'native', device: 'cpu' }
      })
    ]
  ])
  try {
    const fitting = pipeline.fit(X, y)
    assert(fitting instanceof Promise)
    await fitting
    const loaded = await load(pipeline.save())
    try {
      assert.deepEqual(loaded.predict(X), pipeline.predict(X))
    } finally {
      loaded.dispose()
    }
  } finally {
    pipeline.dispose()
  }
  if (process.env.WLEARN_AUTOML_JS) {
    const { autoFit } = require(process.env.WLEARN_AUTOML_JS)
    const shared = await pgModule.create({ core: 'native', device: 'cpu' })
    let creates = 0
    class SharedSym extends SymbolicRegressor {
      static create(params) {
        creates++
        return SymbolicRegressor.create({ ...params, polygrad: shared })
      }
    }
    let result
    try {
      result = await autoFit(
        [
          {
            name: 'sym',
            cls: SharedSym,
            searchSpace: {},
            params: {
              strategy: 'family',
              backend: 'polygrad',
              terms: 2,
              batchSize: 8,
              population: 8,
              eliteCount: 2,
              generations: 1
            }
          }
        ],
        X,
        y,
        { nIter: 1, cv: 2, task: 'regression', ensemble: false }
      )
      assert.equal(creates, 3)
      assert.equal(shared.stats().jit.liveCount, 0)
      assert(Number.isFinite(result.bestScore))
      assert(result.model.predict(X).every(Number.isFinite))
    } finally {
      result?.model.dispose()
      shared.dispose()
    }
    console.log('AutoML: one caller-owned runtime across 2 folds and final refit passed')
  }
  console.log('family public Polygrad: regression, binary, multiclass and batched polish passed')
}
main().catch(err => {
  console.error(err)
  process.exitCode = 1
})

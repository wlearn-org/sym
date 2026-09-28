'use strict'

const fs = require('fs')
const os = require('os')
const path = require('path')
const { createRequire } = require('module')

const pkgRequire = createRequire(path.resolve(__dirname, '../js/package.json'))

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.stack || err.message}`)
    failed++
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  if (Math.abs(a - b) > tol) throw new Error(msg || `expected ${a} ~ ${b}`)
}

function makeRegression(n) {
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const x0 = -2 + 4 * i / (n - 1)
    const x1 = Math.sin(i * 0.37)
    X.push([x0, x1])
    y.push(2 * x0 - 0.5 * x1 + 1)
  }
  return { X, y }
}

function makeClassification(n) {
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const x0 = -2 + 4 * i / (n - 1)
    const x1 = Math.cos(i * 0.19)
    X.push([x0, x1])
    y.push(x0 + 0.25 * x1 > 0 ? 10 : 20)
  }
  return { X, y }
}

const smallParams = {
  population: 80,
  generations: 45,
  maxNodes: 15,
  maxDepth: 4,
  frontierSize: 8,
  eliteCount: 4,
  tournamentSize: 4,
  validationFraction: 0.2,
  complexityPenalty: 0.0001,
  operatorSet: 'basic'
}

async function main() {
  const {
    loadSym,
    SymbolicRegressor,
    SymbolicClassifier,
    FormulaTransformer,
    FormulaVerifier
  } = require('../js/src/index.js')
  const { load } = pkgRequire('@wlearn/core')

  console.log('\n=== WASM ===')
  await test('loadSym loads WASM module', async () => {
    const wasm = await loadSym()
    assert(wasm && typeof wasm.ccall === 'function', 'missing wasm ccall')
  })

  console.log('\n=== SymbolicRegressor ===')
  await test('fits, predicts, exports formula/frontier, saves and loads', async () => {
    const { X, y } = makeRegression(80)
    const model = await SymbolicRegressor.create({ ...smallParams, seed: 123 })
    model.fit(X, y)
    assert(model.isFitted, 'model not fitted')
    assert(model.nFeatures === 2, `nFeatures ${model.nFeatures}`)
    const score = model.score(X, y)
    assert(score > 0.85, `R2 too low: ${score}`)
    const formula = model.formula()
    assert(formula.nodes.length > 0, 'formula has no nodes')
    const text = model.formula({ format: 'text' })
    assert(text.includes('x'), `formula text missing feature: ${text}`)
    const frontier = model.frontier()
    assert(frontier.length > 0, 'frontier empty')
    for (let i = 1; i < frontier.length; i++) {
      assert(frontier[i].objective >= frontier[i - 1].objective - 1e-12, 'frontier not ordered')
    }

    const bundle = model.save()
    const loaded = await SymbolicRegressor.load(bundle)
    try {
      const p1 = model.predict(X)
      const p2 = loaded.predict(X)
      assert(p1.length === p2.length, 'prediction length mismatch')
      for (let i = 0; i < p1.length; i++) assertClose(p1[i], p2[i], 1e-12, 'load parity')
    } finally {
      loaded.dispose()
    }

    const tmpPath = path.join(os.tmpdir(), `wlearn-sym-${process.pid}.wlrn`)
    const diskBundle = model.save(tmpPath)
    try {
      assert(fs.existsSync(tmpPath), 'save(path) did not write a bundle')
      assert(diskBundle.byteLength === bundle.byteLength, 'save(path) byte length mismatch')
      const loadedFromPath = await SymbolicRegressor.load(tmpPath)
      try {
        assert(loadedFromPath.score(X, y) > 0.85, 'path load score')
      } finally {
        loadedFromPath.dispose()
      }
    } finally {
      fs.rmSync(tmpPath, { force: true })
    }

    const registryLoaded = await load(bundle)
    try {
      assert(registryLoaded.score(X, y) > 0.85, 'registry load score')
    } finally {
      registryLoaded.dispose()
    }

    model.dispose()
  })

  await test('modular search options fit and persist', async () => {
    const { X, y } = makeRegression(96)
    const params = {
      ...smallParams,
      population: 96,
      generations: 30,
      maxNodes: 21,
      maxDepth: 5,
      frontierSize: 10,
      validationFraction: 0,
      islands: 3,
      migrationInterval: 5,
      migrationCount: 1,
      warmupGenerations: 12,
      warmupMinNodes: 7,
      broodSize: 3,
      rowSampleSize: 48,
      localRefineInterval: 6,
      localRefineCount: 3,
      complexityHofSize: 24,
      finalSelector: 'loss',
      complexityBucketWidth: 2,
      seed: 778
    }
    const model = await SymbolicRegressor.create(params)
    model.fit(X, y)
    try {
      assert(model.score(X, y) > 0.75, 'modular search score')
      const loaded = await SymbolicRegressor.load(model.save())
      try {
        const loadedParams = loaded.getParams()
        assert(loadedParams.islands === 3, 'loaded islands param')
        assert(loadedParams.broodSize === 3, 'loaded brood param')
        assert(loadedParams.complexityHofSize === 24, 'loaded HOF param')
        assert(loadedParams.finalSelector === 'loss', 'loaded selector param')
        assert(loaded.score(X, y) > 0.75, 'loaded modular search score')
      } finally {
        loaded.dispose()
      }
    } finally {
      model.dispose()
    }
  })

  await test('default search spaces expose advanced search controls', async () => {
    const commonExpected = {
      population: { type: 'int_uniform', low: 128, high: 1024 },
      generations: { type: 'int_uniform', low: 80, high: 400 },
      maxNodes: { type: 'int_uniform', low: 7, high: 63, condition: { strategy: 'tree' } },
      operatorSet: { type: 'categorical', values: ['full', 'smooth', 'basic'] },
      complexityPenalty: { type: 'log_uniform', low: 1e-5, high: 1e-2 },
      islands: [1, 2, 4],
      broodSize: [1, 2],
      complexityHofSize: [0, 64, 128],
      finalSelector: ['objective', 'loss']
    }
    for (const Model of [SymbolicRegressor, SymbolicClassifier]) {
      const space = Model.defaultSearchSpace()
      for (const [key, spec] of Object.entries(commonExpected)) {
        assert(space[key], `${Model.name} missing ${key}`)
        if (Array.isArray(spec)) {
          assert(space[key].type === 'categorical', `${Model.name} ${key} type`)
          assert(JSON.stringify(space[key].values) === JSON.stringify(spec), `${Model.name} ${key} values`)
        } else {
          assert(JSON.stringify(space[key]) === JSON.stringify(spec), `${Model.name} ${key} spec`)
        }
      }
      for (const key of ['maxNodes', 'broodSize', 'complexityHofSize', 'finalSelector']) assert(space[key].condition.strategy === 'tree', `${key} tree condition`)
      for (const key of ['terms', 'ridge', 'immigrantRate']) assert(space[key].condition.strategy === 'family', `${key} family condition`)
      assert(JSON.stringify(space.strategy.values) === JSON.stringify(['tree', 'family']), 'strategy choices')
      assert(JSON.stringify(space.backend.values) === JSON.stringify(['c']), 'implemented backends')
      assert(JSON.stringify(space.mutationRate) === JSON.stringify({ type: 'uniform', low: 0.15, high: 0.55 }), `${Model.name} mutationRate`)
      assert(JSON.stringify(space.crossoverRate) === JSON.stringify({ type: 'uniform', low: 0.35, high: 0.75 }), `${Model.name} crossoverRate`)
    }
    const transformerSpace = FormulaTransformer.defaultSearchSpace()
    for (const key of ['topK', ...Object.keys(commonExpected)]) {
      assert(transformerSpace[key], `FormulaTransformer missing ${key}`)
    }
    for (const [key, values] of Object.entries({
      islands: [1, 2, 4],
      broodSize: [1, 2],
      complexityHofSize: [0, 64, 128],
      finalSelector: ['objective', 'loss']
    })) {
      assert(JSON.stringify(transformerSpace[key].values) === JSON.stringify(values), `FormulaTransformer ${key} values`)
    }
    for (const key of ['mutationRate', 'crossoverRate']) {
      assert(!transformerSpace[key], `FormulaTransformer should not expose ${key}`)
    }
    assert(SymbolicRegressor.defaultSearchSpace().loss.values.includes('mse'), 'regressor loss space')
    assert(!SymbolicClassifier.defaultSearchSpace().loss, 'classifier should not expose regression loss space')
    assert(!FormulaTransformer.defaultSearchSpace().loss, 'transformer should not expose regression loss space')
  })

  await test('Polygrad finalist execution matches C prediction when local polygrad is available', async () => {
    const { X, y } = makeRegression(48)
    const model = await SymbolicRegressor.create({ ...smallParams, seed: 124 })
    model.fit(X, y)
    try {
      const cPred = model.predict(X)
      let pgPred
      try {
        pgPred = await model.predictPolygrad(X)
      } catch (err) {
        if (err && /Cannot find module 'polygrad'/.test(err.message)) return
        throw err
      }
      assert(pgPred.length === cPred.length, 'polygrad length')
      for (let i = 0; i < cPred.length; i++) {
        assertClose(pgPred[i], cPred[i], 1e-4, `polygrad parity ${i}`)
      }
    } finally {
      model.dispose()
    }
  })

  await test('pg-family engine fits, predicts, and saves as a wlearn regressor bundle', async () => {
    const { X, y } = makeRegression(40)
    let model
    try {
      model = await SymbolicRegressor.create({
        engine: 'pg-family',
        population: 24,
        generations: 2,
        terms: 3,
        frontierSize: 6,
        eliteCount: 4,
        seed: 20260705
      })
    } catch (err) {
      if (err && /Cannot find module 'polygrad'/.test(err.message)) return
      throw err
    }
    await model.fit(X, y)
    try {
      assert(model.capabilities.activeEngine === 'pg-family', 'active pg-family engine')
      const pred = await model.predict(X)
      assert(pred.length === X.length, 'pg-family prediction length')
      for (const v of pred) assert(Number.isFinite(v), 'pg-family finite prediction')
      const text = model.formula({ format: 'text' })
      assert(text.includes('x'), `pg-family formula text: ${text}`)
      assert(model.formula({ format: 'json' }).kind === 'sym.pg-family.formula@1', 'pg-family formula kind')
      const bundle = model.save()
      const loaded = await SymbolicRegressor.load(bundle)
      try {
        const loadedPred = await loaded.predict(X)
        assert(loaded.capabilities.activeEngine === 'pg-family', 'loaded active pg-family engine')
        assert(loadedPred.length === pred.length, 'loaded pg-family length')
        for (let i = 0; i < pred.length; i++) assertClose(loadedPred[i], pred[i], 1e-6, `loaded pg-family parity ${i}`)
      } finally {
        loaded.dispose()
      }
    } finally {
      model.dispose()
    }
  })

  await test('pg-family operator sets and batched polish are persisted', async () => {
    const { X, y } = makeRegression(36)
    let model
    try {
      model = await SymbolicRegressor.create({
        engine: 'pg-family',
        population: 12,
        generations: 2,
        terms: 3,
        eliteCount: 3,
        frontierSize: 4,
        seed: 20260707,
        operators: ['add', 'sub', 'mul'],
        polishBatchSize: 12, polishPasses: 1,
        polygrad: { core: 'wasm', device: 'auto' }
      })
    } catch (err) {
      if (err && /Cannot find module 'polygrad'/.test(err.message)) return
      throw err
    }
    await model.fit(X, y)
    try {
      const formula = model.formula({ format: 'json' })
      assert(formula.terms.every(term => ['add', 'sub', 'mul'].includes(term.op)), 'formula respects operators list')
      // Exact elite evaluation counts are checked by test_family_search.c.
      const loaded = await SymbolicRegressor.load(model.save())
      try {
        const params = loaded.getParams()
        assert(JSON.stringify(params.operators) === JSON.stringify(['add', 'sub', 'mul']), 'operators persisted')
        assert(params.polishBatchSize === 12, 'batched polish persisted')
        assert(!('_opIds' in params), 'internal op ids leaked')
      } finally {
        loaded.dispose()
      }
    } finally {
      model.dispose()
    }
  })

  await test('pg-family accepts caller-owned Polygrad runtime without persisting it', async () => {
    const { X, y } = makeRegression(32)
    let polygrad
    try {
      polygrad = process.env.WLEARN_SYM_POLYGRAD_JS
        ? require(process.env.WLEARN_SYM_POLYGRAD_JS)
        : pkgRequire('polygrad')
    } catch (err) {
      if (err && /Cannot find module 'polygrad'/.test(err.message)) return
      throw err
    }
    const pg = polygrad.create({ core: 'wasm', device: 'auto' })
    let model
    try {
      model = await SymbolicRegressor.create({
        engine: 'pg-family',
        population: 12,
        generations: 1,
        terms: 3,
        eliteCount: 3,
        frontierSize: 4,
        seed: 20260708,
        polygrad: pg
      })
      await model.fit(X, y)
      assert(model.capabilities.activeEngine === 'pg-family', 'active pg-family engine')
      assert(!('polygrad' in model.getParams()), 'runtime leaked into public params')
      const loaded = await SymbolicRegressor.load(model.save())
      try {
        assert(!('polygrad' in loaded.getParams()), 'runtime leaked into loaded params')
        const pred = await loaded.predict(X)
        assert(pred.length === X.length, 'loaded runtime-param model predicts')
      } finally {
        loaded.dispose()
      }
    } finally {
      if (model) model.dispose()
      pg.dispose()
    }
  })

  await test('Polygrad post-fit refinement commits only improving constants', async () => {
    const { X, y } = makeRegression(64)
    const model = await SymbolicRegressor.create({ ...smallParams, seed: 123 })
    model.fit(X, y)
    try {
      const formula = model.formula({ format: 'json' })
      const constIndex = formula.nodes.findIndex(n => n.op === 'const')
      assert(constIndex >= 0, 'formula has no constant node')
      model._setFormulaConstant(0, constIndex, formula.nodes[constIndex].value + 1)
      const damagedScore = model.score(X, y)
      let report
      try {
        report = await model.refinePolygrad(X, y, { epochs: 40, lr: 0.01 })
      } catch (err) {
        if (err && /Cannot find module 'polygrad'/.test(err.message)) return
        throw err
      }
      assert(report.committed, `refinement did not commit: ${report.status}`)
      assert(report.afterLoss <= report.beforeLoss + 1e-10, 'refinement worsened MSE')
      assert(model.score(X, y) >= damagedScore, 'refined score did not recover')
      const loaded = await SymbolicRegressor.load(model.save())
      try {
        assert(loaded.score(X, y) >= damagedScore, 'refined constants did not persist')
      } finally {
        loaded.dispose()
      }
    } finally {
      model.dispose()
    }
  })

  await test('Polygrad refinement supports protected division formulas', async () => {
    const { loadPolygrad, refineFormulaPolygrad } = require('../js/src/polygrad.js')
    let runtime
    try {
      runtime = await loadPolygrad()
    } catch (err) {
      if (err && /Cannot find module 'polygrad'/.test(err.message)) return
      throw err
    }
    const X = [[0], [0.5], [1], [2], [4], [8]]
    const y = X.map(row => 2 / (row[0] + 1))
    const formula = {
      nodes: [
        { op: 'const', value: 1.5 },
        { op: 'var', feature: 0 },
        { op: 'const', value: 0.6 },
        { op: 'add', left: 1, right: 2 },
        { op: 'div', left: 0, right: 3 }
      ]
    }
    const report = await refineFormulaPolygrad(runtime, formula, X, y, 1, { epochs: 2, lr: 0.001 })
    assert(report.status === 'ok', `division refinement failed: ${report.reason || report.status}`)
    assert(Number.isFinite(report.loss), 'division refinement produced non-finite loss')
  })

  console.log('\n=== SymbolicClassifier ===')
  await test('binary classifier supports class labels and probabilities', async () => {
    const { X, y } = makeClassification(90)
    const model = await SymbolicClassifier.create({ ...smallParams, seed: 321 })
    model.fit(X, y)
    const acc = model.score(X, y)
    assert(acc > 0.85, `accuracy too low: ${acc}`)
    assert(model.classes[0] === 10 && model.classes[1] === 20, `classes ${model.classes}`)
    const pred = model.predict(X)
    assert(pred.every(v => v === 10 || v === 20), 'predicted original class labels')
    const proba = model.predictProba(X)
    assert(proba.length === X.length * 2, 'proba shape')
    for (let i = 0; i < X.length; i++) assertClose(proba[i * 2] + proba[i * 2 + 1], 1, 1e-10, 'proba sum')
    const loaded = await SymbolicClassifier.load(model.save())
    try {
      assert(loaded.classes[0] === 10 && loaded.classes[1] === 20, 'loaded classes')
      assert(loaded.score(X, y) > 0.85, 'loaded classifier score')
    } finally {
      loaded.dispose()
    }
    model.dispose()
  })

  await test('pg-family classifier supports labels, probabilities, and bundle load', async () => {
    const { X, y } = makeClassification(36)
    let model
    try {
      model = await SymbolicClassifier.create({
        engine: 'pg-family',
        population: 20,
        generations: 2,
        terms: 3,
        frontierSize: 4,
        eliteCount: 4,
        seed: 20260706,
        polygrad: { core: 'wasm', device: 'auto' }
      })
    } catch (err) {
      if (err && /Cannot find module 'polygrad'/.test(err.message)) return
      throw err
    }
    await model.fit(X, y)
    try {
      assert(model.classes[0] === 10 && model.classes[1] === 20, `pg-family classes ${model.classes}`)
      const pred = await model.predict(X)
      assert(pred.length === X.length, 'pg-family classifier prediction length')
      const proba = await model.predictProba(X)
      assert(proba.length === X.length * 2, 'pg-family classifier proba shape')
      for (let i = 0; i < X.length; i++) assertClose(proba[i * 2] + proba[i * 2 + 1], 1, 1e-8, 'pg-family proba sum')
      const loaded = await SymbolicClassifier.load(model.save())
      try {
        const loadedProba = await loaded.predictProba(X)
        assert(loaded.classes[0] === 10 && loaded.classes[1] === 20, 'loaded pg-family classes')
        assert(loadedProba.length === proba.length, 'loaded pg-family proba shape')
      } finally {
        loaded.dispose()
      }
    } finally {
      model.dispose()
    }
  })

  console.log('\n=== FormulaTransformer ===')
  await test('transformer emits top-K formula features', async () => {
    const { X, y } = makeRegression(64)
    const model = await FormulaTransformer.create({ ...smallParams, topK: 4, seed: 555 })
    model.fit(X, y)
    const z = model.transform(X)
    assert(z.length === X.length * 4, `transform length ${z.length}`)
    for (const v of z) assert(Number.isFinite(v), 'finite transform value')
    const loaded = await FormulaTransformer.load(model.save())
    try {
      assert(loaded.transform(X).length === z.length, 'loaded transform length')
    } finally {
      loaded.dispose()
    }
    model.dispose()
  })

  console.log('\n=== FormulaVerifier ===')
  await test('verifier proves affine monotonicity and protected domains', async () => {
    const formula = {
      nodes: [
        { op: 'var', opId: 1, left: -1, right: -1, feature: 0, value: 0 },
        { op: 'const', opId: 0, left: -1, right: -1, feature: -1, value: 2 },
        { op: 'mul', opId: 4, left: 0, right: 1, feature: -1, value: 0 }
      ]
    }
    const report = FormulaVerifier.verify(formula, {
      nFeatures: 1,
      checks: ['domain', { kind: 'monotonicity', feature: 0, direction: 'increasing' }]
    })
    assert(report.status === 'proved', `report status ${report.status}`)
  })

  console.log(`\n${passed} passed, ${failed} failed`)
  if (failed) process.exit(1)
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})

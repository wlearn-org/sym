#!/usr/bin/env node
'use strict'

const fs = require('fs')
const path = require('path')
const { performance } = require('perf_hooks')

const ROOT = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(ROOT, '..')

function parseArgs(argv) {
  const out = {
    datasets: ['friedman1'],
    rows: [128, 512],
    predictRows: 512,
    features: 10,
    noise: 0,
    seed: 20260628,
    population: 32,
    generations: 6,
    maxNodes: 15,
    maxDepth: 4,
    frontierSize: 8,
    operatorSet: 'basic',
    islands: 1,
    migrationInterval: 0,
    migrationCount: 0,
    warmupGenerations: 0,
    warmupMinNodes: 0,
    broodSize: 1,
    rowSampleSize: 0,
    localRefineInterval: 0,
    localRefineCount: 0,
    complexityHofSize: 0,
    finalSelector: 'objective',
    complexityBucketWidth: 2,
    refinePolygrad: false,
    refineEpochs: 80,
    refineLr: 0.001,
    validationFraction: 0,
    xgbNumRound: 200,
    xgbMaxDepth: 4,
    xgbEta: 0.05,
    repeat: 1,
    include: ['sym', 'symc'],
    output: null
  }
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i]
    const val = argv[i + 1]
    if (!key.startsWith('--')) continue
    i++
    const name = key.slice(2)
    if (name === 'dataset') out.datasets = val.split(',').map(v => normalizeDataset(v.trim())).filter(Boolean)
    else if (name === 'datasets') out.datasets = val.split(',').map(v => normalizeDataset(v.trim())).filter(Boolean)
    else if (name === 'rows') out.rows = val.split(',').map(v => Number(v.trim())).filter(Boolean)
    else if (name === 'include') out.include = val.split(',').map(v => v.trim()).filter(Boolean)
    else if (name === 'output') out.output = val
    else if (name === 'operatorSet') out.operatorSet = val
    else if (name === 'finalSelector') out.finalSelector = val
    else if (name === 'refinePolygrad') out.refinePolygrad = val === '1' || val === 'true' || val === 'yes'
    else if (name === 'noise') out.noise = Number(val)
    else if (name in out) out[name] = Number(val)
    else throw new Error(`unknown argument: ${key}`)
  }
  return out
}

function normalizeDataset(value) {
  const key = String(value || '').toLowerCase().replace(/[\s_-]+/g, '')
  if (key === '1' || key === 'friedman' || key === 'friedman1') return 'friedman1'
  if (key === '2' || key === 'friedman2') return 'friedman2'
  if (key === '3' || key === 'friedman3') return 'friedman3'
  throw new Error(`unknown Friedman dataset: ${value}`)
}

function lcg(seed) {
  let s = seed >>> 0
  return function rand() {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0
    return s / 0x100000000
  }
}

function randn(rand) {
  const u1 = Math.max(1e-12, rand())
  const u2 = rand()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

function makeFriedman1(rows, features, seed, noise) {
  if (features < 5) throw new Error('Friedman-1 requires at least 5 features')
  const rand = lcg(seed)
  const X = Array.from({ length: rows }, () => new Array(features).fill(0))
  const y = new Array(rows)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < features; j++) X[i][j] = rand()
    y[i] =
      10 * Math.sin(Math.PI * X[i][0] * X[i][1]) +
      20 * Math.pow(X[i][2] - 0.5, 2) +
      10 * X[i][3] +
      5 * X[i][4] +
      noise * randn(rand)
  }
  return { X, y }
}

function makeFriedman2(rows, seed, noise) {
  const rand = lcg(seed)
  const X = Array.from({ length: rows }, () => new Array(4).fill(0))
  const y = new Array(rows)
  for (let i = 0; i < rows; i++) {
    X[i][0] = rand() * 100
    X[i][1] = rand() * 520 * Math.PI + 40 * Math.PI
    X[i][2] = rand()
    X[i][3] = rand() * 10 + 1
    const inner = X[i][1] * X[i][2] - 1 / (X[i][1] * X[i][3])
    y[i] = Math.sqrt(X[i][0] * X[i][0] + inner * inner) + noise * randn(rand)
  }
  return { X, y }
}

function makeFriedman3(rows, seed, noise) {
  const rand = lcg(seed)
  const X = Array.from({ length: rows }, () => new Array(4).fill(0))
  const y = new Array(rows)
  for (let i = 0; i < rows; i++) {
    X[i][0] = rand() * 100
    X[i][1] = rand() * 520 * Math.PI + 40 * Math.PI
    X[i][2] = rand()
    X[i][3] = rand() * 10 + 1
    const inner = X[i][1] * X[i][2] - 1 / (X[i][1] * X[i][3])
    y[i] = Math.atan(inner / X[i][0]) + noise * randn(rand)
  }
  return { X, y }
}

function makeDataset(dataset, rows, features, seed, noise) {
  if (dataset === 'friedman1') return makeFriedman1(rows, features, seed, noise)
  if (dataset === 'friedman2') return makeFriedman2(rows, seed, noise)
  if (dataset === 'friedman3') return makeFriedman3(rows, seed, noise)
  throw new Error(`unknown dataset: ${dataset}`)
}

function r2(y, pred) {
  const mean = y.reduce((a, b) => a + b, 0) / y.length
  let ssTot = 0
  let ssRes = 0
  for (let i = 0; i < y.length; i++) {
    ssRes += Math.pow(y[i] - pred[i], 2)
    ssTot += Math.pow(y[i] - mean, 2)
  }
  return ssTot <= 1e-24 ? (ssRes <= 1e-24 ? 1 : 0) : 1 - ssRes / ssTot
}

function loadPackage(name) {
  if (name === 'sym') return require(path.join(ROOT, 'js', 'src'))
  if (name === 'symc') return require(path.join(REPO_ROOT, 'symc', 'js', 'src'))
  if (name === 'xgboost') return require(path.join(REPO_ROOT, 'xgboost-wasm', 'src'))
  throw new Error(`unknown package name: ${name}`)
}

async function maybePromise(value) {
  return value && typeof value.then === 'function' ? await value : value
}

async function runOne(name, cfg, dataset, rows, repetition) {
  const pkg = loadPackage(name)
  const train = makeDataset(dataset, rows, cfg.features, cfg.seed + repetition * 100003 + rows, cfg.noise)
  const test = makeDataset(dataset, cfg.predictRows, cfg.features, cfg.seed + repetition * 100003 + rows + 777, cfg.noise)
  const features = train.X[0] ? train.X[0].length : 0
  const isXgb = name === 'xgboost'
  const params = isXgb
    ? {
        task: 'regression',
        objective: 'reg:squarederror',
        numRound: cfg.xgbNumRound,
        max_depth: cfg.xgbMaxDepth,
        eta: cfg.xgbEta,
        subsample: 1,
        colsample_bytree: 1,
        seed: cfg.seed + repetition,
        verbosity: 0
      }
    : {
        population: cfg.population,
        generations: cfg.generations,
        maxNodes: cfg.maxNodes,
        maxDepth: cfg.maxDepth,
        frontierSize: cfg.frontierSize,
        operatorSet: cfg.operatorSet,
        islands: cfg.islands,
        migrationInterval: cfg.migrationInterval,
        migrationCount: cfg.migrationCount,
        warmupGenerations: cfg.warmupGenerations,
        warmupMinNodes: cfg.warmupMinNodes,
        broodSize: cfg.broodSize,
        rowSampleSize: cfg.rowSampleSize,
        localRefineInterval: cfg.localRefineInterval,
        localRefineCount: cfg.localRefineCount,
        complexityHofSize: cfg.complexityHofSize,
        finalSelector: cfg.finalSelector,
        complexityBucketWidth: cfg.complexityBucketWidth,
        validationFraction: cfg.validationFraction,
        complexityPenalty: 0.0001,
        seed: cfg.seed + repetition
      }
  const Model = isXgb ? pkg.XGBModel : pkg.SymbolicRegressor
  const model = await Model.create(params)
  const t0 = performance.now()
  await maybePromise(model.fit(train.X, train.y))
  const fitMs = performance.now() - t0
  let refineMs = 0
  let refineStatus = null
  let refineReason = null
  let refineCommitted = false
  let refineBeforeLoss = null
  let refineAfterLoss = null
  let refineBeforeScore = null
  let refineAfterScore = null
  let refineHistoryFirst = null
  let refineHistoryLast = null
  if (cfg.refinePolygrad && typeof model.refinePolygrad === 'function') {
    const r0 = performance.now()
    try {
      const report = await model.refinePolygrad(train.X, train.y, {
        epochs: cfg.refineEpochs,
        lr: cfg.refineLr
      })
      refineStatus = report.status
      refineReason = report.reason || null
      refineCommitted = !!report.committed
      refineBeforeLoss = Number.isFinite(report.beforeLoss) ? report.beforeLoss : null
      refineAfterLoss = Number.isFinite(report.afterLoss) ? report.afterLoss : null
      refineBeforeScore = Number.isFinite(report.beforeScore) ? report.beforeScore : null
      refineAfterScore = Number.isFinite(report.afterScore) ? report.afterScore : null
      if (Array.isArray(report.history) && report.history.length) {
        refineHistoryFirst = Number(report.history[0])
        refineHistoryLast = Number(report.history[report.history.length - 1])
      }
    } catch (err) {
      refineStatus = `error:${err && err.message ? err.message : err}`
      refineReason = err && err.stack ? err.stack : String(err)
    }
    refineMs = performance.now() - r0
  }

  const t1 = performance.now()
  const pred = await maybePromise(model.predict(test.X))
  const predictMs = performance.now() - t1

  const score = r2(test.y, Array.from(pred))
  const bundleBytes = model.save().byteLength
  const formula = typeof model.formula === 'function' ? model.formula({ format: 'json' }) : null
  const formulaText = typeof model.formula === 'function' ? model.formula({ format: 'text' }) : null
  model.dispose()
  return {
    package: name,
    dataset,
    rows,
    predictRows: cfg.predictRows,
    features,
    population: cfg.population,
    generations: cfg.generations,
    maxNodes: cfg.maxNodes,
    operatorSet: cfg.operatorSet,
    islands: cfg.islands,
    migrationInterval: cfg.migrationInterval,
    migrationCount: cfg.migrationCount,
    warmupGenerations: cfg.warmupGenerations,
    warmupMinNodes: cfg.warmupMinNodes,
    broodSize: cfg.broodSize,
    rowSampleSize: cfg.rowSampleSize,
    localRefineInterval: cfg.localRefineInterval,
    localRefineCount: cfg.localRefineCount,
    complexityHofSize: cfg.complexityHofSize,
    finalSelector: cfg.finalSelector,
    complexityBucketWidth: cfg.complexityBucketWidth,
    refinePolygrad: cfg.refinePolygrad,
    refineMs,
    refineStatus,
    refineReason,
    refineCommitted,
    refineBeforeLoss,
    refineAfterLoss,
    refineBeforeScore,
    refineAfterScore,
    refineHistoryFirst,
    refineHistoryLast,
    repetition,
    fitMs,
    predictMs,
    rowsPerSecondFit: rows / (fitMs / 1000),
    rowsPerSecondPredict: cfg.predictRows / (predictMs / 1000),
    r2: score,
    formulaNodes: formula ? formula.nodes.length : 0,
    formulaText,
    bundleBytes
  }
}

function summarize(results) {
  const groups = new Map()
  for (const row of results) {
    const key = `${row.dataset}|${row.package}|${row.rows}`
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push(row)
  }
  const out = []
  for (const [key, rows] of groups) {
    const [dataset, pkg, n] = key.split('|')
    out.push({
      dataset,
      package: pkg,
      rows: Number(n),
      repeats: rows.length,
      fitMsMean: mean(rows.map(r => r.fitMs)),
      predictMsMean: mean(rows.map(r => r.predictMs)),
      r2Mean: mean(rows.map(r => r.r2)),
        bundleBytesMean: mean(rows.map(r => r.bundleBytes)),
      formulaNodesMean: mean(rows.map(r => r.formulaNodes)),
      refineMsMean: mean(rows.map(r => r.refineMs || 0)),
      refineCommitRate: mean(rows.map(r => r.refineCommitted ? 1 : 0))
    })
  }
  out.sort((a, b) => a.dataset.localeCompare(b.dataset) || a.rows - b.rows || a.package.localeCompare(b.package))
  return out
}

function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1)
}

function printTable(summary) {
  const cols = ['dataset', 'package', 'rows', 'fitMsMean', 'refineMsMean', 'predictMsMean', 'r2Mean', 'refineCommitRate', 'formulaNodesMean', 'bundleBytesMean']
  console.log(cols.join('\t'))
  for (const row of summary) {
    console.log(cols.map(col => {
      const value = row[col]
      return typeof value === 'number' ? value.toFixed(col.endsWith('Mean') || col.endsWith('Rate') ? 3 : 0) : value
    }).join('\t'))
  }
}

async function main() {
  const cfg = parseArgs(process.argv.slice(2))
  const results = []
  for (const dataset of cfg.datasets) {
    for (const rows of cfg.rows) {
      for (const name of cfg.include) {
        for (let rep = 0; rep < cfg.repeat; rep++) {
          const result = await runOne(name, cfg, dataset, rows, rep)
          results.push(result)
          const lossDelta = Number.isFinite(result.refineBeforeLoss) && Number.isFinite(result.refineAfterLoss)
            ? ` loss=${(result.refineAfterLoss - result.refineBeforeLoss).toExponential(3)}`
            : ''
          const refine = cfg.refinePolygrad
            ? ` refine=${result.refineMs.toFixed(1)}ms/${result.refineStatus}/${result.refineCommitted ? 'commit' : 'no'}${lossDelta}${result.refineReason ? ` (${String(result.refineReason).slice(0, 100)})` : ''}`
            : ''
          console.error(`${dataset} ${name} rows=${rows} rep=${rep} fit=${result.fitMs.toFixed(1)}ms${refine} predict=${result.predictMs.toFixed(1)}ms r2=${result.r2.toFixed(4)}`)
        }
      }
    }
  }
  const payload = {
    benchmark: 'friedman-js',
    generatedAt: new Date().toISOString(),
    node: process.version,
    config: cfg,
    summary: summarize(results),
    results
  }
  printTable(payload.summary)
  if (cfg.output) {
    fs.mkdirSync(path.dirname(cfg.output), { recursive: true })
    fs.writeFileSync(cfg.output, JSON.stringify(payload, null, 2))
  }
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err)
  process.exit(1)
})

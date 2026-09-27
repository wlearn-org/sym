'use strict'

// Small held-out comparison of distinct search algorithms, not a scorer-only benchmark.
const fs = require('node:fs')
const os = require('node:os')
const { performance } = require('node:perf_hooks')
const { SymbolicRegressor } = require('../js/src')
function dataset(kind) {
  let state = 1729
  const uniform = () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296 }
  const X = [], y = []
  for (let i = 0; i < 384; i++) {
    const row = Array.from({ length: 5 }, () => uniform())
    const [a, b, c, d, e] = row
    const signal = kind === 'linear' ? 2 * a - b + .5 * c
      : kind === 'smooth' ? Math.sin(3 * a) + b * c + d ** 2
        : 10 * Math.sin(Math.PI * a * b) + 20 * (c - .5) ** 2 + 10 * d + 5 * e
    X.push(row); y.push(signal + .1 * (2 * uniform() - 1))
  }
  return { trainX: X.slice(0, 256), trainY: y.slice(0, 256), testX: X.slice(256), testY: y.slice(256) }
}
async function main() {
  const rows = []
  const params = { population: 32, generations: 6, terms: 4, eliteCount: 4, frontierSize: 4, operatorSet: 'full', validationFraction: 0, complexityPenalty: 1e-5 }
  for (const datasetName of ['linear', 'smooth', 'friedman1']) {
    const data = dataset(datasetName)
    for (const seed of [42, 43, 44]) for (const path of ['family-c-immigrants20', 'family-c-default', 'family-c-islands4-immigrants20', 'legacy-pg-family']) {
      const start = performance.now()
      const model = await SymbolicRegressor.create({ ...params, seed, ...(path !== 'legacy-pg-family' ? { strategy: 'family', backend: 'c', islands: path === 'family-c-islands4-immigrants20' ? 4 : 1, immigrantRate: path === 'family-c-default' ? 0 : .2 } : { engine: 'pg-family' }) })
      const createMs = performance.now() - start
      try {
        const fitStart = performance.now()
        await model.fit(data.trainX, data.trainY)
        const fitMs = performance.now() - fitStart
        const predStart = performance.now()
        await model.predict(data.testX)
        const predictMs = performance.now() - predStart
        const r2 = await model.score(data.testX, data.testY)
        rows.push({ dataset: datasetName, seed, path, createMs, fitMs, predictMs, r2, bundleBytes: model.save().byteLength, searchParams: model.getParams() })
        console.error(JSON.stringify(rows.at(-1)))
      } finally { model.dispose() }
    }
  }
  const report = { description: 'Distinct C and legacy Polygrad searches; same population/generation/term budgets, different proposals and readout solvers. Small diagnostic, not broad performance evidence.', environment: { node: process.version, cpu: os.cpus()[0].model, platform: os.platform() }, params, trainRows: 256, testRows: 128, rows }
  const output = process.argv[2]
  if (output) fs.writeFileSync(output, JSON.stringify(report, null, 2)+'\n')
  else console.log(JSON.stringify(report, null, 2))
}
main().catch(e => { console.error(e); process.exitCode = 1 })

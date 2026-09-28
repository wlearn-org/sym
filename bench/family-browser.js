'use strict'
// Complete browser fits, comparing WASM C and public WebGPU with explicit runtime ownership.
const fs = require('node:fs')
const path = require('node:path')
const http = require('node:http')
const { launchBrowser } = require('../test/test-browser')
async function main() {
  const opts = {
    rows: 512,
    population: 64,
    generations: 10,
    terms: 4,
    batch: 64,
    seed: 11,
    output: null
  }
  for (let i = 2; i < process.argv.length; i += 2) {
    const key = process.argv[i].replace(/^--/, '')
    if (!(key in opts) || process.argv[i + 1] === undefined)
      throw new Error(`invalid option ${key}`)
    opts[key] = key === 'output' ? process.argv[i + 1] : Number(process.argv[i + 1])
  }
  if (!opts.output) throw new Error('--output is required')
  const server = http.createServer((req, res) => res.end('<!doctype html><html></html>'))
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const browser = await launchBrowser()
  try {
    const page = await browser.newPage()
    await page.goto(`http://127.0.0.1:${server.address().port}`)
    await page.addScriptTag({ path: path.resolve(__dirname, '../js/dist/sym.js') })
    const pgRoot = path.dirname(
      require.resolve('polygrad/package.json', { paths: [path.resolve(__dirname, '../js')] })
    )
    await page.addScriptTag({ path: path.join(pgRoot, 'dist/polygrad.async.js') })
    const result = await page.evaluate(async opts => {
      let state = (opts.seed + 1000) >>> 0
      const rand = () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32
      const dataset = n => {
        const X = Array.from({ length: n }, () =>
          Array.from({ length: 5 }, () => Math.fround(rand()))
        )
        const y = X.map(
          x => 10 * Math.sin(Math.PI * x[0] * x[1]) + 20 * (x[2] - 0.5) ** 2 + 10 * x[3] + 5 * x[4]
        )
        return [X, y]
      }
      const [X, y] = dataset(opts.rows),
        [Xt, yt] = dataset(512)
      const base = {
        strategy: 'family',
        terms: opts.terms,
        population: opts.population,
        generations: opts.generations,
        eliteCount: 8,
        validationFraction: 0.2,
        seed: opts.seed,
        polishPasses: 2,
        polishBatchSize: 32,
        batchSize: opts.batch
      }
      const results = []
      let reference
      const fit = async (phase, params, runtimeMs = 0) => {
        const start = performance.now()
        const model = await sym.SymbolicRegressor.create(params)
        try {
          await model.fit(X, y)
          const fitMs = performance.now() - start
          const pred = model.predict(Xt)
          const mean = yt.reduce((a, b) => a + b, 0) / yt.length
          const r2 =
            1 -
            pred.reduce((s, p, i) => s + (p - yt[i]) ** 2, 0) /
              yt.reduce((s, v) => s + (v - mean) ** 2, 0)
          if (!reference) reference = pred
          const delta = Math.max(...Array.from(pred, (p, i) => Math.abs(p - reference[i])))
          if (!Number.isFinite(r2) || !Number.isFinite(delta))
            throw new Error('nonfinite benchmark result')
          results.push({
            phase,
            fit_ms: fitMs,
            runtime_create_ms: runtimeMs,
            total_ms: fitMs + runtimeMs,
            test_r2: r2,
            max_prediction_delta_from_c: delta
          })
        } finally {
          model.dispose()
        }
      }
      await fit('wasm-c', { ...base, backend: 'c' })
      const start = performance.now()
      const runtime = await polygrad.createAsync({ core: 'wasm', device: 'webgpu' })
      const runtimeMs = performance.now() - start
      const device = runtime.device,
        caps = runtime.caps
      try {
        const params = { ...base, backend: 'polygrad', polygrad: runtime, scorerDtype: 'float32' }
        await fit('webgpu-fresh', params, runtimeMs)
        await fit('webgpu-warm', params)
      } finally {
        runtime.dispose()
      }
      return {
        environment: { userAgent: navigator.userAgent, device, caps },
        settings: opts,
        dataset: 'Friedman1, LCG seed+1000; sequential independent train/test rows',
        timing:
          'Complete create+fit including first Sym WASM startup. Fresh includes public Polygrad runtime creation; warm reuses runtime; both rebuild scorers. Same process/driver caches may persist.',
        results
      }
    }, opts)
    fs.mkdirSync(path.dirname(path.resolve(opts.output)), { recursive: true })
    fs.writeFileSync(opts.output, JSON.stringify(result, null, 2) + '\n')
    console.log(JSON.stringify(result, null, 2))
  } finally {
    await browser.close()
    await new Promise(resolve => server.close(resolve))
  }
}
main().catch(e => {
  console.error(e)
  process.exitCode = 1
})

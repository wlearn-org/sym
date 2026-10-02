#!/usr/bin/env node

const path = require('path')
const fs = require('fs')
const http = require('node:http')
const DEVICE = (process.env.POLY_DEV || process.env.DEV || 'cpu').toLowerCase()

const ROOT = path.resolve(__dirname, '..', 'js')
const CORE_PLAYWRIGHT = path.resolve(__dirname, '..', '..', 'wlearn', 'node_modules', 'playwright')
const pkg = require(path.join(ROOT, 'package.json'))
const NAME = pkg.name.split('/').pop()
const EXPORTS = Object.keys(require(path.join(ROOT, 'src', 'index.js')))

function loadChromium() {
  try {
    return require(process.env.WLEARN_PLAYWRIGHT || CORE_PLAYWRIGHT).chromium
  } catch (_) {
    return require('playwright').chromium
  }
}

function executableExists(file) {
  try {
    fs.accessSync(file, fs.constants.X_OK)
    return true
  } catch (_) {
    return false
  }
}

function chromiumExecutablePath(chromium) {
  if (
    process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH &&
    executableExists(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH)
  ) {
    return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
  }
  const expected = chromium.executablePath()
  if (executableExists(expected)) return expected
  const cacheRoot = process.env.PLAYWRIGHT_BROWSERS_PATH || '/opt/ms-playwright'
  try {
    const candidates = fs
      .readdirSync(cacheRoot)
      .filter(name => /^chromium-\d+$/.test(name))
      .sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1]))
      .map(name => path.join(cacheRoot, name, 'chrome-linux64', 'chrome'))
      .filter(executableExists)
    if (candidates.length) return candidates[0]
  } catch (_) {}
  return undefined
}

async function runIifeTest(page, bundle, exportKeys, origin) {
  await page.goto(origin)
  await page.setContent('<!DOCTYPE html><html><body></body></html>', { waitUntil: 'load' })
  await page.addScriptTag({ path: path.join(ROOT, bundle.file) })
  await page.evaluate(
    ({ globalName, exports, device }) => {
      window.__testResult = (async () => {
        try {
          const lib = window[globalName]
          const missing = exports.filter(k => !lib || !(k in lib))
          if (missing.length) return { ok: false, error: 'missing exports: ' + missing.join(', ') }
          const model = await lib.SymbolicRegressor.create({
            population: 12,
            generations: 2,
            maxNodes: 9,
            maxDepth: 3,
            frontierSize: 4,
            seed: 7,
            operatorSet: 'basic'
          })
          const X = [
            [-1, 0],
            [0, 1],
            [1, 0],
            [2, -1],
            [-2, 1],
            [3, 0]
          ]
          const y = [-1, 0.5, 3, 5.5, -3.5, 7]
          await model.fit(X, y)
          const p = await model.predict(X)
          const f = model.formula({ format: 'text' })
          const ok =
            p.length === X.length && f.length > 0 && Array.prototype.every.call(p, Number.isFinite)
          for (const classes of [0, 2, 3]) {
            const Model = classes ? lib.SymbolicClassifier : lib.SymbolicRegressor
            const target = classes ? X.map((_, i) => i % classes) : y
            const family = await Model.create({
              strategy: 'family',
              backend: 'c',
              population: 12,
              generations: 2,
              terms: 2,
              eliteCount: 2,
              seed: 17
            })
            let loaded
            try {
              if (family.fit(X, target) !== family) throw new Error('C family fit must be sync')
              loaded = await Model.load(family.save())
              const before = classes ? family.predictProba(X) : family.predict(X)
              const after = classes ? loaded.predictProba(X) : loaded.predict(X)
              if (!before.every((v, i) => Number.isFinite(v) && v === after[i]))
                throw new Error('C family roundtrip differs')
            } finally {
              loaded?.dispose()
              family.dispose()
            }
          }
          const pg = await lib.SymbolicRegressor.create({
            strategy: 'family',
            backend: 'polygrad',
            hierarchical: true,
            population: 8,
            generations: 2,
            terms: 2,
            eliteCount: 2,
            polishPasses: 1,
            polishBatchSize: 8,
            localRefineInterval: 1,
            localRefineCount: 1,
            batchSize: 8,
            seed: 11,
            operatorSet: 'basic',
            polygrad: { core: 'wasm', device }
          })
          await pg.fit(X, y)
          const pgPred = Array.from(pg.predict(X))
          const pgLoaded = await lib.SymbolicRegressor.load(pg.save())
          const pgPred2 = Array.from(pgLoaded.predict(X))
          const devicePred = await pg.predictPolygrad(X)
          const report = await pg.refinePolygrad(X, y, { epochs: 2, lr: 0.001 })
          const pgOk =
            pgPred.length === X.length &&
            pgPred.every(Number.isFinite) &&
            pgPred.every((v, i) => Math.abs(v - pgPred2[i]) < 1e-5) &&
            pgPred.every((v, i) => Math.abs(v - devicePred[i]) < 1e-5) &&
            report.status === 'ok' &&
            pg.formula({ format: 'json' }).kind === 'sym.family.formula@3'
          pgLoaded.dispose()
          pg.dispose()
          model.dispose()
          return { ok: ok && pgOk, exports: exports.length, formula: f, pg: pgPred.slice(0, 2) }
        } catch (e) {
          return { ok: false, error: e.message, stack: e.stack }
        }
      })()
    },
    { globalName: bundle.global, exports: exportKeys, device: DEVICE }
  )
}

async function runEsmTest(page, bundle, exportKeys) {
  await page.setContent('<!DOCTYPE html><html><body></body></html>', { waitUntil: 'load' })
  const bundleCode = fs.readFileSync(path.join(ROOT, bundle.file), 'utf8')
  const checks = exportKeys
    .map(k => `${JSON.stringify(k)}: typeof ${k} !== 'undefined'`)
    .join(',\n          ')
  await page.addScriptTag({
    type: 'module',
    content: `${bundleCode}
window.__testResult = (async () => {
  try {
    const checks = {
          ${checks}
    }
    const missing = Object.keys(checks).filter(k => !checks[k])
    if (missing.length) return { ok: false, error: 'missing exports: ' + missing.join(', ') }
    return { ok: true, exports: Object.keys(checks).length }
  } catch (e) {
    return { ok: false, error: e.message, stack: e.stack }
  }
})()
`
  })
}

async function launchBrowser() {
  const chromium = loadChromium()
  return chromium.launch({
    headless: DEVICE !== 'webgpu',
    ignoreDefaultArgs: DEVICE === 'webgpu' ? ['--enable-unsafe-swiftshader'] : [],
    executablePath: chromiumExecutablePath(chromium),
    args: [
      '--disable-dev-shm-usage',
      '--no-sandbox',
      '--disable-setuid-sandbox',
      ...(DEVICE === 'webgpu'
        ? [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
            '--disable-gpu-sandbox',
            '--ignore-gpu-blocklist',
            '--disable-software-rasterizer',
            '--use-angle=vulkan'
          ]
        : [])
    ]
  })
}

async function main() {
  const bundles = [
    { name: 'IIFE', file: `dist/${NAME}.js`, type: 'iife', global: NAME },
    { name: 'ESM', file: `dist/${NAME}.mjs`, type: 'esm' }
  ]
  const server = http.createServer((req, res) => res.end('<!doctype html><html></html>'))
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  server.unref()
  const origin = `http://127.0.0.1:${server.address().port}`
  const browser = await launchBrowser()
  let failed = 0
  let passed = 0
  try {
    for (const b of bundles) {
      const page = await browser.newPage()
      if (b.type === 'iife') await runIifeTest(page, b, EXPORTS, origin)
      else await runEsmTest(page, b, EXPORTS)
      await page.waitForFunction(() => window.__testResult, { timeout: 30000 })
      const result = await page.evaluate(() => window.__testResult)
      if (result && result.ok) {
        console.log(`  PASS: ${b.name}`)
        passed++
      } else {
        console.log(`  FAIL: ${b.name} -- ${result ? result.error : 'no result'}`)
        if (result?.stack) console.log(result.stack)
        failed++
      }
      await page.close()
    }
  } finally {
    await browser.close()
    await new Promise(resolve => server.close(resolve))
  }
  console.log(`${passed} browser tests passed, ${failed} failed`)
  if (failed) process.exit(1)
}

if (require.main === module)
  main().catch(err => {
    console.error(err)
    process.exit(1)
  })

module.exports = { launchBrowser }

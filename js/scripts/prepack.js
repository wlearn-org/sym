'use strict'

const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const pkgDir = path.resolve(__dirname, '..')
const pkg = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf8'))
const baseName = pkg.name.split('/').pop()
const skipBuild = /^(1|true|yes)$/i.test(process.env.WLEARN_SKIP_BUILD || '')

function run(cmd, args, extraEnv = {}) {
  const result = spawnSync(cmd, args, {
    cwd: pkgDir,
    stdio: 'inherit',
    env: { ...process.env, ...extraEnv }
  })
  if (result.status !== 0) process.exit(result.status || 1)
}

const buildEnv = {}

// A checkout owns src/; an unpacked package already contains its generated copy.
if (fs.existsSync(path.join(pkgDir, '..', 'src'))) {
  run(process.execPath, ['scripts/sync-csrc.js'])
}

if ((pkg.files || []).includes('wasm/')) {
  const wasmDir = path.join(pkgDir, 'wasm')
  const hasWasm = fs.existsSync(wasmDir) && fs.readdirSync(wasmDir).some(name => name.endsWith('.js'))
  if (skipBuild && !hasWasm) {
    console.error('prepack: WLEARN_SKIP_BUILD=1 but Wasm build is missing')
    process.exit(1)
  }
  if (!skipBuild) run('npm', ['run', 'build'], buildEnv)
}

if ((pkg.files || []).some(entry => entry.startsWith('dist/'))) {
  const distDir = path.join(pkgDir, 'dist')
  const wantJs = path.join(distDir, `${baseName}.js`)
  const wantMjs = path.join(distDir, `${baseName}.mjs`)
  if (skipBuild) {
    if (!fs.existsSync(wantJs) || !fs.existsSync(wantMjs)) {
      console.error('prepack: WLEARN_SKIP_BUILD=1 but browser dist files are missing')
      process.exit(1)
    }
  } else {
    run('npm', ['run', 'build:browser'], buildEnv)
  }
}

for (const entry of pkg.files || []) {
  const full = path.join(pkgDir, entry.replace(/\/$/, ''))
  if (!fs.existsSync(full)) {
    console.error(`prepack: missing published path ${entry}`)
    process.exit(1)
  }
}

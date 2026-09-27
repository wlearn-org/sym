'use strict'

const fs = require('fs')
const { spawnSync } = require('child_process')

const skipBuild = /^(1|true|yes)$/i.test(process.env.WLEARN_SKIP_BUILD || '')

function run(cmd, args, extraEnv = {}) {
  const result = spawnSync(cmd, args, {
    cwd: require('path').resolve(__dirname, '..'),
    stdio: 'inherit',
    env: { ...process.env, ...extraEnv }
  })
  if (result.status !== 0) process.exit(result.status || 1)
}

function findPython() {
  if (process.env.EMSDK_PYTHON) return process.env.EMSDK_PYTHON
  if (process.env.WLEARN_PYTHON) return process.env.WLEARN_PYTHON
  if (process.env.PYTHON) return process.env.PYTHON
  return fs.existsSync('/usr/bin/python3') ? '/usr/bin/python3' : 'python3'
}

const env = { EMSDK_PYTHON: findPython() }
if (!skipBuild) run('npm', ['run', 'build'], env)
run('npm', ['test'], env)

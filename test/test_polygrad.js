'use strict'
const assert = require('node:assert/strict')
const { test } = require('node:test')
const { createRequire } = require('node:module')
const pkgRequire = createRequire(require('node:path').resolve(__dirname, '../js/package.json'))
const { BackendError } = pkgRequire('@wlearn/core')
const { loadPolygrad } = require('../js/src/polygrad')

test('incompatible borrowed runtime fails before tensor allocation without disposal', async () => {
  let disposed = false
  const runtime = {
    Tensor: class {},
    dispose() {
      disposed = true
    }
  }
  await assert.rejects(
    loadPolygrad(runtime),
    error => error instanceof BackendError && /requires.*Polygrad 0\.6/.test(error.message)
  )
  assert.equal(disposed, false)
})

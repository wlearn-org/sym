'use strict'

let wasmModule = null
let loading = null

async function loadSym(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const createSym = require('../wasm/sym.js')
    wasmModule = await createSym(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadSym() first')
  return wasmModule
}

module.exports = { loadSym, getWasm }

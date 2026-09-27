'use strict'

const assert = require('node:assert/strict')
const { loadSym } = require('../js/src/wasm.js')
const { values } = require('./search-abi.json')

async function main() {
  const w = await loadSym()
  const n = 48
  const xp = w._malloc(n * 16)
  const yp = w._malloc(n * 8)
  const scores = w._malloc(64 * 3 * 8)
  let state = 0
  let direct = 0
  let stepped = 0
  try {
    for (let i = 0; i < n; i++) {
      const a = Math.sin(i * 0.37)
      w.HEAPF64[xp / 8 + 2 * i] = a
      w.HEAPF64[xp / 8 + 2 * i + 1] = i / 19
      w.HEAPF64[yp / 8 + i] = a * i / 19 + 0.2
    }
    const args = [xp, n, 2, yp, ...values]
    direct = w._wl_sym_fit(...args)
    state = w._wl_sym_search_new(...args)
    assert.ok(direct && state)
    assert.equal(w._wl_sym_search_batch_id(state), 0)
    let count
    let batches = 0
    while ((count = w._wl_sym_search_propose(state)) > 0) {
      const id = w._wl_sym_search_batch_id(state)
      assert.equal(w._wl_sym_search_propose(state), count)
      assert.equal(w._wl_sym_search_batch_id(state), id)
      assert.ok(w._wl_sym_search_batch_rows(state) > 0)
      assert.ok(w._wl_sym_search_batch_X(state))
      assert.ok(w._wl_sym_search_batch_target(state))
      const nodes = w._wl_sym_search_batch_node_count(state, 0)
      assert.ok(nodes > 0)
      const ints = w._malloc(nodes * 16)
      const constants = w._malloc(nodes * 8)
      try {
        assert.equal(w._wl_sym_search_batch_nodes(state, 0, ints, constants, nodes - 1), -1)
        assert.equal(w._wl_sym_search_batch_nodes(state, 0, ints, constants, nodes), nodes)
        for (let i = 0; i < nodes; i++) {
          const op = w.HEAP32[ints / 4 + 4 * i]
          assert.ok(op >= 0 && op <= 15)
          assert.ok(Number.isFinite(w.HEAPF64[constants / 8 + i]))
        }
      } finally {
        w._free(ints)
        w._free(constants)
      }
      assert.equal(w._wl_sym_search_score(state, id, scores, count), 0)
      assert.equal(w._wl_sym_search_accept(state, id + 1, scores, count), -1)
      assert.equal(w._wl_sym_search_accept(state, id, scores, count), 0)
      assert.equal(w._wl_sym_search_batch_id(state), 0)
      batches++
    }
    assert.equal(count, 0)
    assert.ok(batches > 1)
    stepped = w._wl_sym_search_finish(state)
    assert.ok(stepped)
    function raw(handle) {
      const out = w._malloc(8)
      let data = 0
      try {
        assert.equal(w._wl_sym_save(handle, out, out + 4), 0)
        data = w.HEAP32[out / 4]
        return w.HEAPU8.slice(data, data + w.HEAP32[out / 4 + 1])
      } finally {
        if (data) w._wl_sym_free_buffer(data)
        w._free(out)
      }
    }
    assert.deepEqual(raw(stepped), raw(direct))
    console.log(`Wasm step ABI: ${batches} batches; exact artifact parity`)
  } finally {
    if (state) w._wl_sym_search_free(state)
    if (direct) w._wl_sym_free(direct)
    if (stepped) w._wl_sym_free(stepped)
    w._free(xp)
    w._free(yp)
    w._free(scores)
  }
}
main().catch(error => { console.error(error); process.exitCode = 1 })

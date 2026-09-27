'use strict'

const assert = require('node:assert/strict')
const { loadSym } = require('../js/src/wasm')

async function main() {
  const w = await loadSym()
  const pointers = []
  const alloc = bytes => {
    const p = w._malloc(bytes)
    assert.ok(p)
    pointers.push(p)
    return p
  }
  const xp = alloc(48 * 2 * 8),
    yp = alloc(48 * 8),
    cp = alloc(17 * 8)
  const shape = alloc(8 * 4),
    descriptors = alloc(7 * 3 * 5 * 8)
  const features = alloc(7 * 17 * 3 * 8),
    reference = alloc(7 * 17 * 3 * 8)
  const out = alloc(8)
  function raw(m) {
    assert.equal(w._sym_family_save(m, out, out + 4), 0)
    const p = w.HEAP32[out / 4] >>> 0
    try {
      return w.HEAPU8.slice(p, p + w.HEAP32[out / 4 + 1])
    } finally {
      w._wl_sym_free_buffer(p)
    }
  }
  try {
    for (const classes of [0, 2, 3])
      for (const polish of [0, 1]) {
        for (let r = 0; r < 48; r++) {
          w.HEAPF64[xp / 8 + 2 * r] = (r - 24) / 8
          w.HEAPF64[xp / 8 + 2 * r + 1] = ((r % 7) - 3) / 4
          w.HEAPF64[yp / 8 + r] = classes ? r % classes : r / 8 + (r % 7) / 16
        }
        w.HEAPF64.set(
          [16, 3, 3, 4, 1, 4, 15, 42, polish, 0, 0.2, 1e-5, 0, 0.35, 0.55, 1e-8, 1e-12],
          cp / 8
        )
        const args = [xp, 48, 2, yp, classes > 0 ? 1 : 0, classes, cp, 17]
        const direct = w._wl_sym_family_fit(...args)
        const state = w._wl_sym_family_search_new(...args, 7, 17)
        let stepped = 0,
          status,
          batches = 0
        assert.ok(direct && state)
        try {
          while ((status = w._wl_sym_family_search_propose(state)) > 0) {
            const id = w._wl_sym_family_batch_id(state)
            assert.equal(w._wl_sym_family_search_propose(state), 1)
            assert.equal(w._wl_sym_family_batch_id(state), id)
            assert.equal(w._wl_sym_family_batch_shape(state, shape, 8), 0)
            const [, count, rows, cols, terms, start] = w.HEAP32.slice(shape / 4, shape / 4 + 8)
            const size = count * rows * terms
            assert.equal(
              w._wl_sym_family_batch_descriptors(state, descriptors, count * terms * 5),
              0
            )
            for (let c = 0; c < count; c++)
              for (let r = 0; r < rows; r++)
                for (let t = 0; t < terms; t++) {
                  const at = descriptors / 8 + (c * terms + t) * 5
                  const [a, b, op, p0, p1] = w.HEAPF64.slice(at, at + 5)
                  assert.equal(p0, 1)
                  assert.equal(p1, 0)
                  const x = w.HEAPF64[xp / 8 + (start + r) * cols + a]
                  const y = w.HEAPF64[xp / 8 + (start + r) * cols + b]
                  const value =
                    op === 0
                      ? x + y
                      : op === 1
                      ? x - y
                      : op === 2
                      ? x * y
                      : x / (Math.abs(y) < 1e-6 ? (y >= 0 ? 1e-6 : -1e-6) : y)
                  w.HEAPF64[features / 8 + (c * rows + r) * terms + t] = value
                }
            assert.equal(w._sym_family_search_score(state, id, reference, size), 0)
            assert.deepEqual(
              w.HEAPF64.slice(features / 8, features / 8 + size),
              w.HEAPF64.slice(reference / 8, reference / 8 + size)
            )
            assert.equal(w._sym_family_search_accept(state, id + 1, features, size), -1)
            assert.equal(w._sym_family_search_accept(state, id, features, size - 1), -1)
            assert.equal(w._sym_family_search_accept(state, id, features, size), 0)
            assert.equal(w._sym_family_search_accept(state, id, features, size), -1)
            batches++
          }
          assert.equal(status, 0)
          stepped = w._sym_family_search_finish(state)
          assert.ok(stepped && batches > 1)
          assert.deepEqual(raw(stepped), raw(direct))
          console.log(
            `Wasm family external evaluator: classes=${classes}, polish=${polish}, ${batches} tiles; exact bytes`
          )
        } finally {
          w._sym_family_search_free(state)
          w._sym_family_free(direct)
          if (stepped) w._sym_family_free(stepped)
        }
      }
  } finally {
    for (const p of pointers) w._free(p)
  }
}
main().catch(e => {
  console.error(e)
  process.exitCode = 1
})

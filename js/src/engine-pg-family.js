'use strict'

const { loadPolygrad } = require('./polygrad.js')
const { OPS, resolveOperatorIds, formulaText } = require('./family-format.js')

function lcg(seed) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0
    return s / 0x100000000
  }
}

function randInt(rng, n) {
  return Math.floor(rng() * n)
}

function randRange(rng, lo, hi) {
  return lo + (hi - lo) * rng()
}

function normalizeX(X, nFeatures = 0) {
  if (X && typeof X === 'object' && Number.isFinite(X.rows) && Number.isFinite(X.cols) && X.data) {
    const rows = Number(X.rows)
    const cols = Number(X.cols)
    if (nFeatures && cols !== nFeatures) throw new Error(`X has ${cols} columns, expected ${nFeatures}`)
    return { rows, cols, data: Float32Array.from(X.data) }
  }
  if (X instanceof Float32Array || X instanceof Float64Array) {
    if (!nFeatures) throw new Error('nFeatures is required for flat typed-array input')
    if (X.length % nFeatures !== 0) throw new Error('flat X length is not divisible by nFeatures')
    return { rows: X.length / nFeatures, cols: nFeatures, data: Float32Array.from(X) }
  }
  if (!Array.isArray(X)) throw new Error('X must be rows or a flat typed array')
  const rows = X.length
  const cols = rows > 0 && Array.isArray(X[0]) ? X[0].length : 1
  const data = new Float32Array(rows * cols)
  for (let r = 0; r < rows; r++) {
    const row = Array.isArray(X[r]) ? X[r] : [X[r]]
    if (row.length !== cols) throw new Error(`X row ${r} has ${row.length} columns, expected ${cols}`)
    for (let c = 0; c < cols; c++) {
      const v = Number(row[c])
      if (!Number.isFinite(v)) throw new Error(`X[${r},${c}] must be finite`)
      data[r * cols + c] = v
    }
  }
  return { rows, cols, data }
}

function normalizeY(y, rows) {
  const out = y instanceof Float32Array ? new Float32Array(y) : Float32Array.from(y)
  if (out.length !== rows) throw new Error(`y length ${out.length} does not match X rows ${rows}`)
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i])) throw new Error(`y[${i}] must be finite`)
  }
  return out
}

function flatten(value, out = []) {
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    for (const item of value) out.push(item)
    return out
  }
  if (Array.isArray(value)) {
    for (const item of value) flatten(item, out)
  } else {
    out.push(value)
  }
  return out
}

async function tensorToFloat32(t) {
  if (t && typeof t.toTypedArray === 'function') {
    try {
      return Float32Array.from(await t.toTypedArray())
    } catch (err) {
      if (!t.toTypedArrayAsync) throw err
      return Float32Array.from(await t.toTypedArrayAsync())
    }
  }
  let arr
  try {
    arr = await t.toArray()
  } catch (err) {
    if (!t.toArrayAsync) throw err
    arr = await t.toArrayAsync()
  }
  return Float32Array.from(flatten(arr))
}

async function tensorsToFloat32(tensors, Tensor) {
  if (!Array.isArray(tensors)) return [await tensorToFloat32(tensors)]
  if (Tensor && typeof Tensor.toTypedArrays === 'function') {
    let arrays
    try {
      arrays = await Tensor.toTypedArrays(...tensors)
    } catch (err) {
      if (!Tensor.toTypedArraysAsync) throw err
      arrays = await Tensor.toTypedArraysAsync(...tensors)
    }
    return arrays.map(arr => Float32Array.from(arr))
  }
  const arrays = []
  for (const t of tensors) arrays.push(await tensorToFloat32(t))
  return arrays
}

function protectedDiv(a, b) {
  const eps = 1e-6
  const den = Math.abs(b) < eps ? (b >= 0 ? eps : -eps) : b
  return a / den
}

function applyOp(op, a, b, p0, p1) {
  const z = a * p0 + p1
  if (op === 0) return a + b
  if (op === 1) return a - b
  if (op === 2) return a * b
  if (op === 3) return protectedDiv(a, b)
  if (op === 4) return Math.sin(z)
  if (op === 5) return Math.cos(z)
  if (op === 6) return Math.tanh(z)
  if (op === 7) return Math.log(Math.abs(z) + 1e-6)
  if (op === 8) return Math.sqrt(Math.abs(z))
  return Math.exp(Math.max(-6, Math.min(6, z)))
}

function candidateKey(c) {
  const parts = []
  for (let t = 0; t < c.terms; t++) {
    parts.push(c.fa[t], c.fb[t], c.op[t], Math.round(c.p0[t] * 1000), Math.round(c.p1[t] * 1000))
  }
  return parts.join(':')
}

function evaluationKey(c) {
  const parts = []
  for (let t = 0; t < c.terms; t++) {
    parts.push(
      c.fa[t],
      c.fb[t],
      c.op[t],
      Number(c.p0[t]).toPrecision(9),
      Number(c.p1[t]).toPrecision(9)
    )
  }
  return parts.join(':')
}

function cloneCandidate(c) {
  return {
    terms: c.terms,
    fa: Int32Array.from(c.fa),
    fb: Int32Array.from(c.fb),
    op: Int32Array.from(c.op),
    p0: Float32Array.from(c.p0),
    p1: Float32Array.from(c.p1),
    coef: Float32Array.from(c.coef),
    bias: c.bias,
    loss: c.loss,
    objective: c.objective,
    complexity: c.complexity,
    generation: c.generation
  }
}

function randomCandidate(rng, nFeatures, terms, opIds = null) {
  const ops = opIds && opIds.length ? opIds : OPS.map((_, i) => i)
  const c = {
    terms,
    fa: new Int32Array(terms),
    fb: new Int32Array(terms),
    op: new Int32Array(terms),
    p0: new Float32Array(terms),
    p1: new Float32Array(terms),
    coef: new Float32Array(terms),
    bias: randRange(rng, -0.5, 0.5),
    loss: Infinity,
    objective: Infinity,
    complexity: terms + 1,
    generation: 0
  }
  for (let t = 0; t < terms; t++) {
    c.fa[t] = randInt(rng, nFeatures)
    c.fb[t] = randInt(rng, nFeatures)
    c.op[t] = ops[randInt(rng, ops.length)]
    c.p0[t] = randRange(rng, -2, 2)
    c.p1[t] = randRange(rng, -1, 1)
    c.coef[t] = randRange(rng, -1, 1)
  }
  return c
}

function complexity(c) {
  let v = 1
  for (let t = 0; t < c.terms; t++) {
    v += c.op[t] >= 4 ? 3 : 2
    if (Math.abs(c.p0[t] - 1) > 1e-6) v += 1
    if (Math.abs(c.p1[t]) > 1e-6) v += 1
  }
  return v
}

function mutateCandidate(parent, rng, nFeatures, params) {
  const opIds = params._opIds || OPS.map((_, i) => i)
  const c = cloneCandidate(parent)
  const t = randInt(rng, c.terms)
  const r = rng()
  if (r < 0.20) c.fa[t] = randInt(rng, nFeatures)
  else if (r < 0.40) c.fb[t] = randInt(rng, nFeatures)
  else if (r < 0.58) c.op[t] = opIds[randInt(rng, opIds.length)]
  else if (r < 0.78) c.p0[t] = Math.max(-4, Math.min(4, c.p0[t] + randRange(rng, -0.4, 0.4)))
  else if (r < 0.94) c.p1[t] = Math.max(-4, Math.min(4, c.p1[t] + randRange(rng, -0.4, 0.4)))
  else {
    const q = randInt(rng, c.terms)
    c.fa[q] = randInt(rng, nFeatures)
    c.fb[q] = randInt(rng, nFeatures)
    c.op[q] = opIds[randInt(rng, opIds.length)]
    c.p0[q] = randRange(rng, -2, 2)
    c.p1[q] = randRange(rng, -1, 1)
  }
  if (rng() < (params.constantMutationRate ?? 0.25)) {
    const q = randInt(rng, c.terms)
    c.p0[q] = Math.max(-4, Math.min(4, c.p0[q] + randRange(rng, -0.2, 0.2)))
    c.p1[q] = Math.max(-4, Math.min(4, c.p1[q] + randRange(rng, -0.2, 0.2)))
  }
  c.loss = Infinity
  c.objective = Infinity
  return c
}

function crossover(a, b, rng) {
  const c = cloneCandidate(a)
  for (let t = 0; t < c.terms; t++) {
    if (rng() < 0.5) {
      c.fa[t] = b.fa[t]
      c.fb[t] = b.fb[t]
      c.op[t] = b.op[t]
      c.p0[t] = b.p0[t]
      c.p1[t] = b.p1[t]
    }
  }
  c.loss = Infinity
  c.objective = Infinity
  return c
}

function solveLinear(aFlat, b, n, ridge) {
  const a = new Float64Array(aFlat)
  const rhs = new Float64Array(b)
  for (let i = 0; i < n; i++) a[i * n + i] += ridge
  for (let k = 0; k < n; k++) {
    let pivot = k
    let best = Math.abs(a[k * n + k])
    for (let r = k + 1; r < n; r++) {
      const v = Math.abs(a[r * n + k])
      if (v > best) { best = v; pivot = r }
    }
    if (best < 1e-12) return null
    if (pivot !== k) {
      for (let j = k; j < n; j++) {
        const tmp = a[k * n + j]
        a[k * n + j] = a[pivot * n + j]
        a[pivot * n + j] = tmp
      }
      const br = rhs[k]
      rhs[k] = rhs[pivot]
      rhs[pivot] = br
    }
    const diag = a[k * n + k]
    for (let j = k; j < n; j++) a[k * n + j] /= diag
    rhs[k] /= diag
    for (let r = 0; r < n; r++) {
      if (r === k) continue
      const f = a[r * n + k]
      if (Math.abs(f) < 1e-20) continue
      for (let j = k; j < n; j++) a[r * n + j] -= f * a[k * n + j]
      rhs[r] -= f * rhs[k]
    }
  }
  return rhs
}

function r2(y, pred) {
  let mean = 0
  for (const v of y) mean += v
  mean /= Math.max(1, y.length)
  let ssTot = 0
  let ssRes = 0
  for (let i = 0; i < y.length; i++) {
    const e = y[i] - pred[i]
    const d = y[i] - mean
    ssRes += e * e
    ssTot += d * d
  }
  return ssTot <= 1e-24 ? (ssRes <= 1e-24 ? 1 : 0) : 1 - ssRes / ssTot
}

function candidateToPlain(c) {
  return {
    terms: c.terms,
    fa: Array.from(c.fa),
    fb: Array.from(c.fb),
    op: Array.from(c.op),
    p0: Array.from(c.p0),
    p1: Array.from(c.p1),
    coef: Array.from(c.coef),
    bias: c.bias,
    loss: c.loss,
    objective: c.objective,
    complexity: c.complexity,
    generation: c.generation
  }
}

function candidateFromPlain(o) {
  return {
    terms: o.terms,
    fa: Int32Array.from(o.fa),
    fb: Int32Array.from(o.fb),
    op: Int32Array.from(o.op),
    p0: Float32Array.from(o.p0),
    p1: Float32Array.from(o.p1),
    coef: Float32Array.from(o.coef),
    bias: Number(o.bias),
    loss: Number(o.loss ?? Infinity),
    objective: Number(o.objective ?? Infinity),
    complexity: Number(o.complexity ?? o.terms + 1),
    generation: Number(o.generation ?? 0)
  }
}

function applyCachedEvaluation(c, cached) {
  c.bias = cached.bias
  c.coef.set(cached.coef)
  c.loss = cached.loss
  c.objective = cached.objective
  c.complexity = cached.complexity
}

function rememberEvaluation(cache, c, maxSize) {
  if (!cache || !maxSize || maxSize <= 0 || !Number.isFinite(c.objective)) return
  cache.set(evaluationKey(c), {
    bias: c.bias,
    coef: Float32Array.from(c.coef),
    loss: c.loss,
    objective: c.objective,
    complexity: c.complexity
  })
  while (cache.size > maxSize) {
    const first = cache.keys().next().value
    cache.delete(first)
  }
}

function serializableParams(params) {
  const {
    polygrad,
    polygradRuntime,
    ownPolygradRuntime,
    _opIds,
    wasm,
    ...rest
  } = params || {}
  return { ...rest }
}

class PgFamilyRegressorEngine {
  constructor(params = {}) {
    this.params = {
      population: 128,
      generations: 20,
      terms: 6,
      eliteCount: 8,
      tournamentSize: 4,
      frontierSize: 16,
      complexityPenalty: 1e-5,
      ridge: 1e-8,
      seed: 42,
      polygrad: null,
      tensorDevice: null,
      gatherMode: 'selector',
      scoreMode: 'summary',
      stackSummaries: 'auto',
      evalCacheSize: 8192,
      ...params
    }
    this._opIds = resolveOperatorIds(this.params)
    this.params._opIds = this._opIds
    this.pg = params.polygradRuntime || null
    this.ownsRuntime = Boolean(params.ownPolygradRuntime)
    this.best = null
    this.archive = []
    this.nFeatures = 0
    this.featureNames = []
    this.fitted = false
    this.stats = null
    this._summaryJit = null
    this._lossJit = null
    this._predictJit = null
  }

  static async create(params = {}) {
    const model = new PgFamilyRegressorEngine(params)
    if (!model.pg) {
      model.pg = await loadPolygrad(model.params.polygrad)
      model.ownsRuntime = !(model.params.polygrad && model.params.polygrad.Tensor)
    }
    return model
  }

  _tensorOptions(dtype = 'float32') {
    const opts = { dtype }
    const requested = this.params.tensorDevice ||
      (this.params.polygrad && this.params.polygrad.device !== 'auto' ? this.params.polygrad.device : null)
    if (requested) opts.device = requested
    return opts
  }

  _tensor(data, dtype = 'float32') {
    return new this.pg.Tensor(data, this._tensorOptions(dtype))
  }

  _makeInputs(population, matrix, y = null, coeffs = true, cache = null) {
    const inputs = {
      cols: matrix.cols,
      y: cache && cache.y ? cache.y : (y ? this._tensor(y) : this._tensor(new Float32Array(matrix.rows))),
      bias: this._tensor(Float32Array.from(population, c => c.bias)),
      x: null,
      selA: [],
      selB: [],
      idxA: [],
      idxB: [],
      slotA: [],
      slotB: [],
      p0: [],
      p1: [],
      coef: [],
      mask: [],
      opIds: this._opIds
    }
    const gatherMode = this.params.gatherMode || 'selector'
    const useGather = gatherMode === 'gather'
    const useSelector = gatherMode === 'selector'
    if (!useGather && !useSelector && gatherMode !== 'slot') {
      throw new Error(`unsupported gatherMode: ${gatherMode}`)
    }
    if (useGather && this.pg.canRun) {
      try {
        if (!this.pg.canRun({ op: 'gather', shape: [2, 2] })) {
          throw new Error('Polygrad runtime reports gather is not runnable')
        }
      } catch (err) {
        if (!String(err && err.message || err).includes('cannot prove')) throw err
      }
    }
    if (useGather && !this.pg.Tensor.prototype.gather) {
      throw new Error('gatherMode="gather" requires a Polygrad runtime with Tensor.gather')
    }
    if (useSelector || useGather) {
      inputs.x = cache && cache.x ? cache.x : this._tensor(matrix.data)
    }
    for (let t = 0; t < this.params.terms; t++) {
      const slotA = (useSelector || useGather) ? null : new Float32Array(population.length * matrix.rows)
      const slotB = (useSelector || useGather) ? null : new Float32Array(population.length * matrix.rows)
      const selA = useSelector ? new Float32Array(population.length * matrix.cols) : null
      const selB = useSelector ? new Float32Array(population.length * matrix.cols) : null
      const idxA = useGather ? new Int32Array(matrix.rows * population.length) : null
      const idxB = useGather ? new Int32Array(matrix.rows * population.length) : null
      const p0 = new Float32Array(population.length)
      const p1 = new Float32Array(population.length)
      const coef = new Float32Array(population.length)
      const masks = this._opIds.map(() => new Float32Array(population.length))
      for (let c = 0; c < population.length; c++) {
        const cand = population[c]
        p0[c] = cand.p0[t]
        p1[c] = cand.p1[t]
        coef[c] = coeffs ? cand.coef[t] : 0
        const pos = this._opIds.indexOf(cand.op[t])
        if (pos < 0) throw new Error(`candidate uses operator outside configured pg-family operators: ${OPS[cand.op[t]] || cand.op[t]}`)
        masks[pos][c] = 1
        const fa = cand.fa[t]
        const fb = cand.fb[t]
        if (useSelector) {
          selA[c * matrix.cols + fa] = 1
          selB[c * matrix.cols + fb] = 1
        } else if (useGather) {
          for (let r = 0; r < matrix.rows; r++) {
            const off = r * population.length + c
            idxA[off] = fa
            idxB[off] = fb
          }
        } else {
          const off = c * matrix.rows
          for (let r = 0; r < matrix.rows; r++) {
            slotA[off + r] = matrix.data[r * matrix.cols + fa]
            slotB[off + r] = matrix.data[r * matrix.cols + fb]
          }
        }
      }
      if (useSelector) {
        inputs.selA.push(this._tensor(selA))
        inputs.selB.push(this._tensor(selB))
      } else if (useGather) {
        inputs.idxA.push(this._tensor(idxA, 'int32'))
        inputs.idxB.push(this._tensor(idxB, 'int32'))
      } else {
        inputs.slotA.push(this._tensor(slotA))
        inputs.slotB.push(this._tensor(slotB))
      }
      inputs.p0.push(this._tensor(p0))
      inputs.p1.push(this._tensor(p1))
      inputs.coef.push(this._tensor(coef))
      inputs.mask.push(masks.map(mask => this._tensor(mask)))
    }
    return inputs
  }

  async _fitNeeded(population, matrix, y, cache = null, evalCache = null) {
    const pending = []
    let cacheHits = 0
    for (const cand of population) {
      if (Number.isFinite(cand.objective) && Number.isFinite(cand.loss)) {
        rememberEvaluation(evalCache, cand, this.params.evalCacheSize)
        continue
      }
      const cached = evalCache ? evalCache.get(evaluationKey(cand)) : null
      if (cached) {
        applyCachedEvaluation(cand, cached)
        cacheHits++
      } else {
        pending.push(cand)
      }
    }
    if (pending.length) {
      await this._fitCoefficients(pending, matrix, y, cache)
      for (const cand of pending) rememberEvaluation(evalCache, cand, this.params.evalCacheSize)
    }
    return { fitCount: pending.length, cacheHits }
  }

  _branchForOp(op, a, b, z) {
    if (op === 0) return a.add(b)
    if (op === 1) return a.sub(b)
    if (op === 2) return a.mul(b)
    if (op === 3) {
      const safeDen = b.abs().lt(1e-6).where(b.ge(0).where(1e-6, -1e-6), b)
      return a.div(safeDen)
    }
    if (op === 4) return z.sin()
    if (op === 5) return z.cos()
    if (op === 6) return z.tanh()
    if (op === 7) return z.abs().add(1e-6).log()
    if (op === 8) return z.abs().sqrt()
    return z.clamp(-6, 6).exp()
  }

  _termOutputs(inputs, rows, candidates) {
    const { Tensor } = this.pg
    const out = []
    const xRows = inputs.idxA.length ? inputs.x.reshape(rows, inputs.cols) : null
    const xT = inputs.x && !inputs.idxA.length ? inputs.x.reshape(rows, inputs.cols).transpose() : null
    for (let t = 0; t < this.params.terms; t++) {
      let a
      let b
      if (xRows) {
        a = xRows.gather(1, inputs.idxA[t].reshape(rows, candidates)).transpose()
        b = xRows.gather(1, inputs.idxB[t].reshape(rows, candidates)).transpose()
      } else if (xT) {
        a = inputs.selA[t].reshape(candidates, inputs.cols).dot(xT)
        b = inputs.selB[t].reshape(candidates, inputs.cols).dot(xT)
      } else {
        a = inputs.slotA[t].reshape(candidates, rows)
        b = inputs.slotB[t].reshape(candidates, rows)
      }
      const p0 = inputs.p0[t].reshape(candidates, 1)
      const p1 = inputs.p1[t].reshape(candidates, 1)
      const z = a.mul(p0).add(p1)
      let term = Tensor.zeros(candidates, rows, this._tensorOptions())
      for (let pos = 0; pos < inputs.opIds.length; pos++) {
        const op = inputs.opIds[pos]
        term = term.add(inputs.mask[t][pos].reshape(candidates, 1).mul(this._branchForOp(op, a, b, z)))
      }
      out.push(term)
    }
    return out
  }

  _summaryGraph(inputs, rows, candidates) {
    const { Tensor } = this.pg
    const terms = this._termOutputs(inputs, rows, candidates)
    const y = inputs.y.reshape(1, rows)
    const outs = []
    for (let t = 0; t < terms.length; t++) outs.push(terms[t].sum(1))
    for (let t = 0; t < terms.length; t++) outs.push(terms[t].mul(y).sum(1))
    for (let t = 0; t < terms.length; t++) {
      for (let s = t; s < terms.length; s++) outs.push(terms[t].mul(terms[s]).sum(1))
    }
    return this.params.stackSummaries === false ? outs : Tensor.stack(outs, { dim: 0 })
  }

  _lossGraph(inputs, rows, candidates) {
    const terms = this._termOutputs(inputs, rows, candidates)
    const y = inputs.y.reshape(1, rows)
    let pred = inputs.bias.reshape(candidates, 1)
    for (let t = 0; t < terms.length; t++) {
      pred = pred.add(inputs.coef[t].reshape(candidates, 1).mul(terms[t]))
    }
    return pred.sub(y).square().mean(1)
  }

  _predictGraph(inputs, rows) {
    return this._lossOrPredictGraph(inputs, rows, 1, true)
  }

  _lossOrPredictGraph(inputs, rows, candidates, predictOnly) {
    const terms = this._termOutputs(inputs, rows, candidates)
    let pred = inputs.bias.reshape(candidates, 1)
    for (let t = 0; t < terms.length; t++) {
      pred = pred.add(inputs.coef[t].reshape(candidates, 1).mul(terms[t]))
    }
    return predictOnly ? pred.reshape(rows) : pred
  }

  _makeTensorCache(matrix, y) {
    return {
      x: this._tensor(matrix.data),
      y: this._tensor(y)
    }
  }

  async _fitCoefficients(population, matrix, y, cache = null) {
    const inputs = this._makeInputs(population, matrix, y, false, cache)
    const fn = bundle => this._summaryGraph(bundle, matrix.rows, population.length)
    let tensors
    const useJit = this.params.jit !== false && typeof this.pg.jit === 'function' && population.length === this.params.population
    if (!useJit) tensors = fn(inputs)
    else {
      if (!this._summaryJit) this._summaryJit = this.pg.jit(fn, { prune: true })
      tensors = await this._summaryJit(inputs)
    }
    let arrays
    if (Array.isArray(tensors)) {
      arrays = await tensorsToFloat32(tensors, this.pg.Tensor)
    } else {
      const flat = await tensorToFloat32(tensors)
      arrays = []
      for (let i = 0; i < flat.length; i += population.length) {
        arrays.push(flat.subarray(i, i + population.length))
      }
    }
    const n = this.params.terms + 1
    const sumY = y.reduce((a, b) => a + b, 0)
    let yTy = 0
    for (let i = 0; i < y.length; i++) yTy += y[i] * y[i]
    const yOffset = this.params.terms
    const crossOffset = this.params.terms * 2
    for (let c = 0; c < population.length; c++) {
      const A = new Float64Array(n * n)
      const b = new Float64Array(n)
      A[0] = matrix.rows
      b[0] = sumY
      for (let t = 0; t < this.params.terms; t++) {
        const st = arrays[t][c]
        A[0 * n + t + 1] = st
        A[(t + 1) * n + 0] = st
        b[t + 1] = arrays[yOffset + t][c]
      }
      let k = crossOffset
      for (let t = 0; t < this.params.terms; t++) {
        for (let s = t; s < this.params.terms; s++) {
          const v = arrays[k++][c]
          A[(t + 1) * n + s + 1] = v
          A[(s + 1) * n + t + 1] = v
        }
      }
      const sol = solveLinear(A, b, n, this.params.ridge)
      if (sol) {
        population[c].bias = sol[0]
        for (let t = 0; t < this.params.terms; t++) population[c].coef[t] = sol[t + 1]
        let quad = 0
        let lin = 0
        for (let i = 0; i < n; i++) {
          lin += sol[i] * b[i]
          for (let j = 0; j < n; j++) quad += sol[i] * A[i * n + j] * sol[j]
        }
        population[c].loss = Math.max(0, (quad - 2 * lin + yTy) / Math.max(1, matrix.rows))
        population[c].complexity = complexity(population[c])
        population[c].objective = population[c].loss + this.params.complexityPenalty * population[c].complexity
      } else {
        population[c].loss = Infinity
        population[c].complexity = complexity(population[c])
        population[c].objective = Infinity
      }
    }
  }

  async _score(population, matrix, y, cache = null) {
    const inputs = this._makeInputs(population, matrix, y, true, cache)
    const fn = bundle => this._lossGraph(bundle, matrix.rows, population.length)
    let lossTensor
    const useJit = this.params.jit !== false && typeof this.pg.jit === 'function' && population.length === this.params.population
    if (!useJit) lossTensor = fn(inputs)
    else {
      if (!this._lossJit) this._lossJit = this.pg.jit(fn, { prune: true })
      lossTensor = await this._lossJit(inputs)
    }
    const losses = await tensorToFloat32(lossTensor)
    for (let i = 0; i < population.length; i++) {
      population[i].loss = losses[i]
      population[i].complexity = complexity(population[i])
      population[i].objective = population[i].loss + this.params.complexityPenalty * population[i].complexity
    }
  }

  _scoreCpu(population, matrix, y) {
    for (const cand of population) {
      let sum = 0
      let valid = true
      for (let r = 0; r < matrix.rows; r++) {
        let pred = cand.bias
        for (let t = 0; t < cand.terms; t++) {
          const a = matrix.data[r * matrix.cols + cand.fa[t]]
          const b = matrix.data[r * matrix.cols + cand.fb[t]]
          pred += cand.coef[t] * applyOp(cand.op[t], a, b, cand.p0[t], cand.p1[t])
        }
        if (!Number.isFinite(pred)) {
          valid = false
          break
        }
        const e = pred - y[r]
        sum += e * e
      }
      cand.loss = valid ? sum / Math.max(1, matrix.rows) : Infinity
      cand.complexity = complexity(cand)
      cand.objective = cand.loss + this.params.complexityPenalty * cand.complexity
    }
  }

  _archive(population, generation) {
    const byKey = new Map()
    for (const old of this.archive) byKey.set(candidateKey(old), old)
    for (const cand of population) {
      const c = cloneCandidate(cand)
      c.generation = generation
      const key = candidateKey(c)
      const prev = byKey.get(key)
      if (!prev || c.objective < prev.objective) byKey.set(key, c)
    }
    this.archive = Array.from(byKey.values())
      .sort((a, b) => a.objective - b.objective || a.loss - b.loss || a.complexity - b.complexity)
      .slice(0, this.params.frontierSize)
    this.best = cloneCandidate(this.archive[0])
  }

  _select(population, rng) {
    let best = null
    for (let i = 0; i < this.params.tournamentSize; i++) {
      const cand = population[randInt(rng, population.length)]
      if (!best || cand.objective < best.objective) best = cand
    }
    return best
  }

  async fit(X, y) {
    const matrix = normalizeX(X)
    const target = normalizeY(y, matrix.rows)
    this.nFeatures = matrix.cols
    this.featureNames = Array.from({ length: matrix.cols }, (_, i) => `x${i}`)
    const tensorCache = this._makeTensorCache(matrix, target)
    const evalCache = this.params.evalCacheSize === 0 ? null : new Map()
    const rng = lcg(this.params.seed)
    let population = Array.from(
      { length: this.params.population },
      () => randomCandidate(rng, matrix.cols, this.params.terms, this._opIds)
    )
    const timings = []
    for (let gen = 0; gen < this.params.generations; gen++) {
      const t0 = performance.now()
      const fitReport = await this._fitNeeded(population, matrix, target, tensorCache, evalCache)
      const coeffMs = performance.now() - t0
      const t1 = performance.now()
      if (this.params.scoreMode === 'polygrad') await this._score(population, matrix, target, tensorCache)
      else if (this.params.scoreMode !== 'summary') this._scoreCpu(population, matrix, target)
      const scoreMs = performance.now() - t1
      population.sort((a, b) => a.objective - b.objective)
      this._archive(population, gen)
      const next = population.slice(0, this.params.eliteCount).map(cloneCandidate)
      const seen = new Set(next.map(candidateKey))
      while (next.length < this.params.population) {
        let child
        if (rng() < 0.20) child = randomCandidate(rng, matrix.cols, this.params.terms, this._opIds)
        else if (rng() < 0.55) child = crossover(this._select(population, rng), this._select(population, rng), rng)
        else child = mutateCandidate(this._select(population, rng), rng, matrix.cols, this.params)
        const key = candidateKey(child)
        if (!seen.has(key) || next.length > this.params.population * 0.95) {
          seen.add(key)
          next.push(child)
        }
      }
      population = next
      timings.push({
        generation: gen,
        coeffMs,
        scoreMs,
        fitCount: fitReport.fitCount,
        cacheHits: fitReport.cacheHits,
        bestLoss: this.best.loss,
        bestObjective: this.best.objective
      })
    }
    await this._fitNeeded(this.archive, matrix, target, tensorCache, evalCache)
    if (this.params.scoreMode === 'polygrad') await this._score(this.archive, matrix, target, tensorCache)
    else if (this.params.scoreMode !== 'summary') this._scoreCpu(this.archive, matrix, target)
    this.archive.sort((a, b) => a.objective - b.objective)
    this.best = cloneCandidate(this.archive[0])
    this.fitted = true
    this.stats = {
      runtime: this.pg
        ? {
            core: this.pg.core || 'default',
            device: this.pg.device || 'default',
            caps: this.pg.caps || null
          }
        : null,
      gatherMode: this.params.gatherMode || 'selector',
      timings,
      trainLoss: this.best.loss
    }
    return this
  }

  _predictCpu(matrix, cand = this.best) {
    const out = new Float32Array(matrix.rows)
    for (let r = 0; r < matrix.rows; r++) {
      let pred = cand.bias
      for (let t = 0; t < cand.terms; t++) {
        const a = matrix.data[r * matrix.cols + cand.fa[t]]
        const b = matrix.data[r * matrix.cols + cand.fb[t]]
        pred += cand.coef[t] * applyOp(cand.op[t], a, b, cand.p0[t], cand.p1[t])
      }
      out[r] = pred
    }
    return out
  }

  async predict(X, opts = {}) {
    if (!this.fitted || !this.best) throw new Error('model is not fitted')
    const matrix = normalizeX(X, this.nFeatures)
    if (opts.backend !== 'polygrad') return this._predictCpu(matrix)
    const inputs = this._makeInputs([this.best], matrix, null, true)
    const fn = bundle => this._predictGraph(bundle, matrix.rows)
    try {
      let pred
      if (this.params.jit === false || opts.jit === false || opts.jit == null) pred = fn(inputs)
      else {
        if (!this._predictJit) this._predictJit = this.pg.jit(fn, { prune: true })
        pred = await this._predictJit(inputs)
      }
      return tensorToFloat32(pred)
    } catch (err) {
      if (opts.strictPolygrad) throw err
      return this._predictCpu(matrix)
    }
  }

  async score(X, y) {
    const matrix = normalizeX(X, this.nFeatures)
    const target = normalizeY(y, matrix.rows)
    const pred = await this.predict(matrix.data, { backend: 'cpu' })
    return r2(target, pred)
  }

  formula(opts = {}) {
    if (!this.best) throw new Error('model is not fitted')
    if ((opts.format || 'json') === 'text') return formulaText(this.best, this.featureNames)
    return {
      kind: 'sym.pg-family.formula@1',
      text: formulaText(this.best, this.featureNames),
      terms: Array.from({ length: this.best.terms }, (_, t) => ({
        coefficient: this.best.coef[t],
        op: OPS[this.best.op[t]],
        featureA: this.best.fa[t],
        featureB: this.best.fb[t],
        p0: this.best.p0[t],
        p1: this.best.p1[t]
      })),
      bias: this.best.bias,
      loss: this.best.loss,
      objective: this.best.objective,
      complexity: this.best.complexity
    }
  }

  frontier(opts = {}) {
    return this.archive.map(c => {
      const out = candidateToPlain(c)
      if (opts.format === 'text') out.text = formulaText(c, this.featureNames)
      return out
    })
  }

  getParams() {
    return { ...this.params }
  }

  setParams(params = {}) {
    Object.assign(this.params, params)
    this._opIds = resolveOperatorIds(this.params)
    this.params._opIds = this._opIds
    if (this._summaryJit) this._summaryJit.dispose()
    if (this._lossJit) this._lossJit.dispose()
    if (this._predictJit) this._predictJit.dispose()
    this._summaryJit = null
    this._lossJit = null
    this._predictJit = null
    return this
  }

  save(pathOrUndefined) {
    const bytes = Buffer.from(JSON.stringify(this.toState()))
    if (pathOrUndefined != null) {
      throw new Error('PgFamilyRegressorEngine.save(path) is not part of the product API; save the owning wlearn bundle instead')
    }
    return new Uint8Array(bytes)
  }

  toState() {
    if (!this.fitted || !this.best) throw new Error('model is not fitted')
    const payload = {
      kind: 'wlearn.sym.pg-family.regressor@1',
      params: serializableParams(this.params),
      nFeatures: this.nFeatures,
      featureNames: this.featureNames,
      fitted: this.fitted,
      best: this.best ? candidateToPlain(this.best) : null,
      archive: this.archive.map(candidateToPlain),
      stats: this.stats
    }
    return payload
  }

  static async fromState(payload, params = {}) {
    if (!payload || payload.kind !== 'wlearn.sym.pg-family.regressor@1') {
      throw new Error('not a wlearn sym pg-family regressor state')
    }
    const merged = { ...payload.params, ...params }
    const model = await PgFamilyRegressorEngine.create(merged)
    model.nFeatures = Number(payload.nFeatures || 0)
    model.featureNames = payload.featureNames || Array.from({ length: model.nFeatures }, (_, i) => `x${i}`)
    model.fitted = Boolean(payload.fitted)
    model.best = payload.best ? candidateFromPlain(payload.best) : null
    model.archive = (payload.archive || []).map(candidateFromPlain)
    model.stats = payload.stats || null
    return model
  }

  static fromStateSync(payload, params = {}) {
    if (!payload || payload.kind !== 'wlearn.sym.pg-family.regressor@1') {
      throw new Error('not a wlearn sym pg-family regressor state')
    }
    const model = new PgFamilyRegressorEngine({ ...payload.params, ...params, polygradRuntime: null })
    model.nFeatures = Number(payload.nFeatures || 0)
    model.featureNames = payload.featureNames || Array.from({ length: model.nFeatures }, (_, i) => `x${i}`)
    model.fitted = Boolean(payload.fitted)
    model.best = payload.best ? candidateFromPlain(payload.best) : null
    model.archive = (payload.archive || []).map(candidateFromPlain)
    model.stats = payload.stats || null
    return model
  }

  dispose() {
    if (this._summaryJit) this._summaryJit.dispose()
    if (this._lossJit) this._lossJit.dispose()
    if (this._predictJit) this._predictJit.dispose()
    this._summaryJit = null
    this._lossJit = null
    this._predictJit = null
    if (this.ownsRuntime && this.pg && this.pg.dispose) this.pg.dispose()
    this.pg = null
  }
}

function uniqueSortedFloat(y) {
  return [...new Set(Array.from(y, Number))].sort((a, b) => a - b)
}

function softmaxRows(scores, rows, cols) {
  const out = new Float64Array(rows * cols)
  for (let r = 0; r < rows; r++) {
    let max = -Infinity
    for (let c = 0; c < cols; c++) max = Math.max(max, scores[r * cols + c])
    let sum = 0
    for (let c = 0; c < cols; c++) {
      const v = Math.exp(Math.max(-60, Math.min(60, scores[r * cols + c] - max)))
      out[r * cols + c] = v
      sum += v
    }
    if (sum <= 0 || !Number.isFinite(sum)) {
      const p = 1 / cols
      for (let c = 0; c < cols; c++) out[r * cols + c] = p
    } else {
      for (let c = 0; c < cols; c++) out[r * cols + c] /= sum
    }
  }
  return out
}

class PgFamilyClassifierEngine {
  constructor(params = {}) {
    this.params = { ...params }
    this.classes = []
    this.models = []
    this.nFeatures = 0
    this.fitted = false
    this.stats = null
    this.pg = params.polygradRuntime || null
    this.ownsRuntime = Boolean(params.ownPolygradRuntime)
  }

  static async create(params = {}) {
    return new PgFamilyClassifierEngine(params)
  }

  async fit(X, y) {
    const matrix = normalizeX(X)
    const target = normalizeY(y, matrix.rows)
    this.classes = this.params.classes ? Array.from(this.params.classes, Number) : uniqueSortedFloat(target)
    if (this.classes.length < 2) throw new Error('classification requires at least two classes')
    this.nFeatures = matrix.cols

    if (!this.pg) {
      this.pg = await loadPolygrad(this.params.polygrad)
      this.ownsRuntime = !(this.params.polygrad && this.params.polygrad.Tensor)
    }
    const baseParams = {
      ...this.params,
      polygradRuntime: this.pg
    }
    delete baseParams.classes
    const runtimes = []
    this.models = []

    if (this.classes.length === 2) {
      const bin = new Float32Array(target.length)
      for (let i = 0; i < target.length; i++) bin[i] = Number(target[i]) === this.classes[1] ? 1 : -1
      const model = await PgFamilyRegressorEngine.create(baseParams)
      await model.fit(matrix, bin)
      this.models.push(model)
      runtimes.push(model.stats && model.stats.runtime)
    } else {
      for (const cls of this.classes) {
        const one = new Float32Array(target.length)
        for (let i = 0; i < target.length; i++) one[i] = Number(target[i]) === cls ? 1 : -1
        const model = await PgFamilyRegressorEngine.create(baseParams)
        await model.fit(matrix, one)
        this.models.push(model)
        runtimes.push(model.stats && model.stats.runtime)
      }
    }

    this.fitted = true
    this.stats = { runtimes }
    return this
  }

  _ensureFitted() {
    if (!this.fitted || !this.models.length) throw new Error('model is not fitted')
  }

  async decisionFunction(X) {
    this._ensureFitted()
    const matrix = normalizeX(X, this.nFeatures)
    if (this.classes.length === 2) {
      const score = await this.models[0].predict(matrix.data, { backend: 'cpu' })
      const out = new Float64Array(score.length)
      for (let i = 0; i < score.length; i++) out[i] = score[i]
      return out
    }
    const out = new Float64Array(matrix.rows * this.classes.length)
    for (let c = 0; c < this.classes.length; c++) {
      const pred = await this.models[c].predict(matrix.data, { backend: 'cpu' })
      for (let r = 0; r < matrix.rows; r++) out[r * this.classes.length + c] = pred[r]
    }
    return out
  }

  async predictProba(X) {
    this._ensureFitted()
    const matrix = normalizeX(X, this.nFeatures)
    const scores = await this.decisionFunction(matrix.data)
    if (this.classes.length === 2) {
      const out = new Float64Array(matrix.rows * 2)
      for (let r = 0; r < matrix.rows; r++) {
        const p1 = 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, scores[r]))))
        out[r * 2] = 1 - p1
        out[r * 2 + 1] = p1
      }
      return out
    }
    return softmaxRows(scores, matrix.rows, this.classes.length)
  }

  async predict(X) {
    this._ensureFitted()
    const matrix = normalizeX(X, this.nFeatures)
    const scores = await this.decisionFunction(matrix.data)
    const out = new Float64Array(matrix.rows)
    if (this.classes.length === 2) {
      for (let r = 0; r < matrix.rows; r++) out[r] = scores[r] >= 0 ? this.classes[1] : this.classes[0]
      return out
    }
    for (let r = 0; r < matrix.rows; r++) {
      let best = 0
      let bestScore = scores[r * this.classes.length]
      for (let c = 1; c < this.classes.length; c++) {
        const score = scores[r * this.classes.length + c]
        if (score > bestScore) {
          bestScore = score
          best = c
        }
      }
      out[r] = this.classes[best]
    }
    return out
  }

  async score(X, y) {
    const matrix = normalizeX(X, this.nFeatures)
    const target = normalizeY(y, matrix.rows)
    const pred = await this.predict(matrix.data)
    let correct = 0
    for (let i = 0; i < target.length; i++) if (Number(target[i]) === Number(pred[i])) correct++
    return correct / Math.max(1, target.length)
  }

  formula(opts = {}) {
    this._ensureFitted()
    const index = Number(opts.index ?? 0)
    const model = this.classes.length === 2 ? this.models[0] : this.models[index]
    const item = model.formula(opts)
    if ((opts.format || 'json') === 'text') return item
    return {
      ...item,
      classLabel: this.classes.length === 2 ? this.classes[1] : this.classes[index],
      role: this.classes.length === 2 ? 'binary-margin' : 'one-vs-rest-margin'
    }
  }

  frontier(opts = {}) {
    this._ensureFitted()
    const out = []
    for (let i = 0; i < this.models.length; i++) {
      for (const item of this.models[i].frontier(opts)) {
        out.push({
          ...item,
          classIndex: this.classes.length === 2 ? 1 : i,
          classLabel: this.classes.length === 2 ? this.classes[1] : this.classes[i]
        })
      }
    }
    return out
  }

  toState() {
    this._ensureFitted()
    return {
      kind: 'wlearn.sym.pg-family.classifier@1',
      params: serializableParams(this.params),
      classes: this.classes,
      nFeatures: this.nFeatures,
      fitted: this.fitted,
      models: this.models.map(model => model.toState()),
      stats: this.stats
    }
  }

  static fromStateSync(payload, params = {}) {
    if (!payload || payload.kind !== 'wlearn.sym.pg-family.classifier@1') {
      throw new Error('not a wlearn sym pg-family classifier state')
    }
    const model = new PgFamilyClassifierEngine({ ...payload.params, ...params })
    model.classes = Array.from(payload.classes || [], Number)
    model.nFeatures = Number(payload.nFeatures || 0)
    model.fitted = Boolean(payload.fitted)
    model.models = (payload.models || []).map(state => PgFamilyRegressorEngine.fromStateSync(state))
    model.stats = payload.stats || null
    return model
  }

  dispose() {
    for (const model of this.models) model.dispose()
    this.models = []
    if (this.ownsRuntime && this.pg && this.pg.dispose) this.pg.dispose()
    this.pg = null
    this.fitted = false
  }
}

module.exports = {
  PgFamilyRegressorEngine,
  PgFamilyClassifierEngine,
  OPS,
  r2,
  normalizeX
}

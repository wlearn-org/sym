'use strict'

const { ValidationError } = require('@wlearn/core')
const { getWasm } = require('./wasm.js')
const { OPS, resolveOperatorIds, formulaText } = require('./family-format.js')

function config(p) {
  const get = (a, b, fallback) => p[a] ?? p[b] ?? fallback
  if (p.loss != null && p.loss !== 'mse' && p.loss !== 0)
    throw new ValidationError(
      'family currently supports the MSE readout, including classification margins'
    )
  for (const key of [
    'maxNodes',
    'max_nodes',
    'maxDepth',
    'max_depth',
    'topK',
    'top_k',
    'broodSize',
    'brood_size',
    'rowSampleSize',
    'row_sample_size',
    'complexityHofSize',
    'complexity_hof_size',
    'finalSelector',
    'final_selector',
    'hierarchical',
    'migrationInterval',
    'migration_interval',
    'migrationCount',
    'migration_count',
    'localRefineInterval',
    'local_refine_interval',
    'localRefineCount',
    'local_refine_count',
    'scoreMode',
    'score_mode',
    'jit',
    'gatherMode',
    'gather_mode',
    'polygradRuntime',
    'polygrad',
    'tournamentSize',
    'tournament_size',
    'constantRate',
    'constant_rate',
    'constMin',
    'const_min',
    'constMax',
    'const_max',
    'huberDelta',
    'huber_delta',
    'warmupGenerations',
    'warmup_generations',
    'warmupMinNodes',
    'warmup_min_nodes',
    'complexityBucketWidth',
    'complexity_bucket_width'
  ]) {
    if (p[key] != null) throw new ValidationError(`${key} is not supported by C family search`)
  }
  let mask
  try {
    mask = resolveOperatorIds(p).reduce((v, op) => v | (1 << op), 0)
  } catch (err) {
    throw new ValidationError(err.message)
  }
  const values = [
    p.population ?? 128,
    p.generations ?? 20,
    p.terms ?? 6,
    get('eliteCount', 'elite_count', 8),
    p.islands ?? 1,
    get('frontierSize', 'frontier_size', 16),
    mask,
    p.seed ?? 42,
    get('polishPasses', 'polish_passes', 0),
    get('earlyStopRounds', 'early_stop_rounds', 0),
    get('validationFraction', 'validation_fraction', 0),
    get('complexityPenalty', 'complexity_penalty', 1e-5),
    get('immigrantRate', 'immigrant_rate', 0),
    get('mutationRate', 'mutation_rate', 0.35),
    get('crossoverRate', 'crossover_rate', 0.55),
    p.ridge ?? 1e-8,
    p.tol ?? 1e-12
  ]
  if (values.some(v => typeof v !== 'number' || !Number.isFinite(v)))
    throw new ValidationError('family parameters must be finite numbers')
  return Float64Array.from(values)
}
function error(w) {
  return new ValidationError(w.ccall('wl_sym_get_last_error', 'string', [], []))
}
function withArrays(arrays, fn) {
  const w = getWasm(),
    ptrs = []
  try {
    for (const a of arrays) {
      const ptr = w._malloc(Math.max(8, a.length * 8))
      if (!ptr) throw new Error('family allocation failed')
      ptrs.push(ptr)
      w.HEAPF64.set(a, ptr / 8)
    }
    return fn(w, ...ptrs)
  } finally {
    for (const ptr of ptrs) w._free(ptr)
  }
}
function matrix(X, cols) {
  if (X && X.data && Number.isInteger(X.rows) && Number.isInteger(X.cols)) return X
  if (ArrayBuffer.isView(X)) {
    if (X.length % cols) throw new ValidationError('family matrix shape mismatch')
    return { rows: X.length / cols, cols, data: Float64Array.from(X) }
  }
  if (!Array.isArray(X) || X.some(row => !Array.isArray(row) || row.length !== cols))
    throw new ValidationError('family matrix shape mismatch')
  return { rows: X.length, cols, data: Float64Array.from(X.flat()) }
}
function packed(c) {
  if (!c || !Number.isInteger(c.terms) || c.terms < 1 || c.terms > 32)
    throw new ValidationError('invalid family term count')
  for (const key of ['fa', 'fb', 'op', 'p0', 'p1', 'coef'])
    if (!Array.isArray(c[key]) || c[key].length !== c.terms)
      throw new ValidationError(`invalid family ${key}`)
  const out = []
  for (let i = 0; i < c.terms; i++) out.push(c.fa[i], c.fb[i], c.op[i], c.p0[i], c.p1[i], c.coef[i])
  out.push(c.bias, c.loss, c.validLoss ?? c.loss, c.complexity, c.objective, c.generation ?? 0)
  if (out.some(v => typeof v !== 'number' || !Number.isFinite(v)))
    throw new ValidationError('invalid family numeric value')
  return Float64Array.from(out)
}
class CFamilyEngine {
  constructor(params = {}, classes = []) {
    this.params = { ...params }
    this.classes = [...classes]
    this.handle = 0
    this.nFeatures = 0
    this.terms = 0
    this.artifactVersion = 1
  }
  fit(X, y) {
    const options = config(this.params)
    const index = new Map(this.classes.map((c, i) => [c, i]))
    const target = this.classes.length
      ? Float64Array.from(y, c => {
          if (!index.has(c)) throw new ValidationError(`unknown class ${c}`)
          return index.get(c)
        })
      : y
    const handle = withArrays([X.data, target, options], (w, xp, yp, pp) => {
      const h = w._wl_sym_family_fit(
        xp,
        X.rows,
        X.cols,
        yp,
        this.classes.length ? 1 : 0,
        this.classes.length,
        pp,
        options.length
      )
      if (!h) throw error(w)
      return h
    })
    this.dispose()
    this.handle = handle
    this.nFeatures = X.cols
    this.terms = options[2]
    this.artifactVersion = 2
    return this
  }
  _predict(X, mode) {
    const m = matrix(X, this.nFeatures)
    const heads = this.classes.length > 2 ? this.classes.length : 1
    const cols = mode === 2 ? this.classes.length : mode === 1 ? heads : 1
    if (
      !Number.isInteger(m.rows) ||
      m.rows < 0 ||
      m.rows > 2147483647 ||
      m.cols !== this.nFeatures ||
      m.data.length !== m.rows * m.cols
    )
      throw new ValidationError('family matrix shape mismatch')
    return withArrays([m.data, new Float64Array(m.rows * cols)], (w, xp, out) => {
      if (w._sym_family_predict(this.handle, xp, m.rows, m.cols, mode, out)) throw error(w)
      return w.HEAPF64.slice(out / 8, out / 8 + m.rows * cols)
    })
  }
  predict(X, opts = {}) {
    if (opts.backend === 'polygrad')
      throw new ValidationError('Polygrad execution for C family models is not implemented yet')
    const out = this._predict(X, 0)
    return this.classes.length ? Float64Array.from(out, i => this.classes[i]) : out
  }
  decisionFunction(X) {
    return this._predict(X, 1)
  }
  predictProba(X) {
    return this._predict(X, 2)
  }
  score(X, y) {
    const p = this.predict(X)
    if (y.length !== p.length) throw new ValidationError('target length mismatch')
    if (this.classes.length)
      return p.reduce((sum, v, i) => sum + (v === y[i] ? 1 : 0), 0) / p.length
    const mean = Array.from(y).reduce((a, b) => a + b, 0) / y.length
    let residual = 0,
      total = 0
    for (let i = 0; i < y.length; i++) {
      residual += (y[i] - p[i]) ** 2
      total += (y[i] - mean) ** 2
    }
    return total === 0 ? (residual === 0 ? 1 : 0) : 1 - residual / total
  }
  _candidate(head, index) {
    return withArrays([new Float64Array(6 * this.terms + 6)], (w, ptr) => {
      const n = w._sym_family_export(this.handle, head, index, ptr, 6 * this.terms + 6)
      if (n < 0) throw error(w)
      const values = w.HEAPF64.slice(ptr / 8, ptr / 8 + n)
      const c = { terms: this.terms, fa: [], fb: [], op: [], p0: [], p1: [], coef: [] }
      for (let i = 0; i < this.terms; i++)
        ['fa', 'fb', 'op', 'p0', 'p1', 'coef'].forEach((k, j) => c[k].push(values[6 * i + j]))
      const offset = 6 * this.terms
      ;['bias', 'loss', 'validLoss', 'complexity', 'objective', 'generation'].forEach((k, j) => {
        c[k] = values[offset + j]
      })
      return c
    })
  }
  formula(opts = {}) {
    const head = this.classes.length > 2 ? Number(opts.index ?? 0) : 0
    const c = this._candidate(head, -1),
      names = Array.from({ length: this.nFeatures }, (_, i) => `x${i}`)
    const text = formulaText(c, names)
    if (opts.format === 'text') return text
    const result = {
      kind: 'sym.pg-family.formula@1',
      text,
      bias: c.bias,
      loss: c.loss,
      validLoss: c.validLoss,
      objective: c.objective,
      complexity: c.complexity,
      terms: c.op.map((op, i) => ({
        coefficient: c.coef[i],
        op: OPS[op],
        featureA: c.fa[i],
        featureB: c.fb[i],
        p0: c.p0[i],
        p1: c.p1[i]
      }))
    }
    if (this.classes.length)
      Object.assign(result, {
        classLabel: this.classes[this.classes.length === 2 ? 1 : head],
        role: this.classes.length === 2 ? 'binary-margin' : 'one-vs-rest-margin'
      })
    return result
  }
  frontier(opts = {}) {
    const out = [],
      w = getWasm(),
      heads = this.classes.length > 2 ? this.classes.length : 1
    for (let h = 0; h < heads; h++)
      for (let i = 0; i < w._sym_family_archive_size(this.handle, h); i++) {
        const c = this._candidate(h, i)
        if (opts.format === 'text')
          c.text = formulaText(
            c,
            Array.from({ length: this.nFeatures }, (_, j) => `x${j}`)
          )
        if (this.classes.length) {
          c.classIndex = this.classes.length === 2 ? 1 : h
          c.classLabel = this.classes[c.classIndex]
        }
        out.push(c)
      }
    return out
  }
  toState() {
    const { polygrad, wasm, polygradRuntime, ...params } = this.params
    const heads = this.classes.length > 2 ? this.classes.length : 1
    const models = Array.from({ length: heads }, (_, h) => ({
      kind: 'wlearn.sym.pg-family.regressor@1',
      params,
      nFeatures: this.nFeatures,
      featureNames: Array.from({ length: this.nFeatures }, (_, i) => `x${i}`),
      fitted: true,
      best: this._candidate(h, -1),
      archive: Array.from({ length: getWasm()._sym_family_archive_size(this.handle, h) }, (_, i) =>
        this._candidate(h, i)
      ),
      stats: null
    }))
    return this.classes.length
      ? {
          kind: 'wlearn.sym.pg-family.classifier@1',
          params,
          nFeatures: this.nFeatures,
          fitted: true,
          classes: this.classes,
          models,
          stats: null
        }
      : models[0]
  }
  toBytes() {
    return withArrays([new Float64Array(1), new Float64Array(1)], (w, pp, np) => {
      if (w._sym_family_save(this.handle, pp, np)) throw error(w)
      const ptr = w.HEAP32[pp / 4] >>> 0,
        size = w.HEAP32[np / 4]
      try {
        return w.HEAPU8.slice(ptr, ptr + size)
      } finally {
        w._wl_sym_free_buffer(ptr)
      }
    })
  }
  static fromBytes(bytes, params, classes) {
    const w = getWasm(),
      m = new CFamilyEngine(params, classes)
    const ptr = w._malloc(Math.max(1, bytes.byteLength))
    if (!ptr) throw new ValidationError('family allocation failed')
    try {
      w.HEAPU8.set(bytes, ptr)
      m.handle = w._sym_family_load(ptr, bytes.byteLength)
    } finally {
      w._free(ptr)
    }
    if (!m.handle) throw error(w)
    try {
      const dims = withArrays([new Float64Array(2)], (w, out) => {
        if (w._sym_family_dimensions(m.handle, out)) throw error(w)
        return w.HEAP32.slice(out / 4, out / 4 + 4)
      })
      if (
        !Array.isArray(classes) ||
        classes.length !== dims[1] ||
        classes.some(v => typeof v !== 'number' || !Number.isFinite(v)) ||
        new Set(classes).size !== classes.length
      )
        throw new ValidationError('family artifact classes mismatch')
      m.nFeatures = dims[0]
      m.terms = dims[2]
      m.artifactVersion = 2
      return m
    } catch (err) {
      m.dispose()
      throw err
    }
  }
  static fromState(state, params = {}) {
    const classifier = state?.kind === 'wlearn.sym.pg-family.classifier@1'
    if (!state?.fitted) throw new ValidationError('family payload is not fitted')
    if (!classifier && state?.kind !== 'wlearn.sym.pg-family.regressor@1')
      throw new ValidationError('invalid family payload kind')
    const classes = classifier ? state.classes : []
    if (
      !Array.isArray(classes) ||
      (classifier && (classes.length < 2 || classes.length > 128)) ||
      classes.some(v => typeof v !== 'number' || !Number.isFinite(v)) ||
      new Set(classes).size !== classes.length
    )
      throw new ValidationError('invalid family classes')
    const models = classifier ? state.models : [state],
      heads = classes.length > 2 ? classes.length : 1
    if (
      !Array.isArray(models) ||
      models.length !== heads ||
      models.some(
        m =>
          m?.kind !== 'wlearn.sym.pg-family.regressor@1' ||
          !m?.fitted ||
          m.nFeatures !== state.nFeatures ||
          !Array.isArray(m.archive)
      )
    )
      throw new ValidationError('invalid family heads')
    const terms = models[0].best?.terms,
      frontier = Math.max(1, ...models.map(m => m.archive.length))
    if (
      !Number.isInteger(state.nFeatures) ||
      state.nFeatures < 1 ||
      state.nFeatures > 2147483647 ||
      !Number.isInteger(terms) ||
      terms < 1 ||
      terms > 32 ||
      frontier > 128
    )
      throw new ValidationError('invalid family dimensions')
    const m = new CFamilyEngine({ ...state.params, ...params }, classes),
      w = getWasm()
    m.handle = w._sym_family_new(state.nFeatures, classes.length, terms, frontier)
    if (!m.handle) throw error(w)
    m.nFeatures = state.nFeatures
    m.terms = terms
    try {
      models.forEach((head, h) =>
        [head.best, ...head.archive].forEach((c, i) => {
          if (c?.terms !== terms) throw new ValidationError('family head term count mismatch')
          withArrays([packed(c)], (w, ptr) => {
            if (w._sym_family_import(m.handle, h, i - 1, ptr, 6 * terms + 6)) throw error(w)
          })
        })
      )
      return m
    } catch (err) {
      m.dispose()
      throw err
    }
  }
  dispose() {
    if (this.handle) getWasm()._sym_family_free(this.handle)
    this.handle = 0
  }
}
module.exports = { CFamilyEngine }

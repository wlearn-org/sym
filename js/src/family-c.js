'use strict'

const { ValidationError, normalizeX } = require('@wlearn/core')
const { getWasm } = require('./wasm.js')
const { OPS, resolveOperatorIds, formulaText } = require('./family-format.js')

function config(p) {
  const get = (a, b, fallback) => p[a] ?? p[b] ?? fallback
  if (p.loss != null && p.loss !== 'mse' && p.loss !== 0)
    throw new ValidationError(
      'family currently supports the MSE readout, including classification margins'
    )
  for (const key of [
    'stackSummaries',
    'evalCacheSize',
    'tensorDevice',
    'ownPolygradRuntime',
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
    'migrationInterval',
    'migration_interval',
    'migrationCount',
    'migration_count',
    'scoreMode',
    'score_mode',
    'jit',
    'gatherMode',
    'gather_mode',
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
  const scaled = get('scaleAware', 'scale_aware', false)
  const method = get('polishMethod', 'polish_method', 'coordinate')
  if (typeof scaled !== 'boolean' || !['coordinate', 'lm'].includes(method))
    throw new ValidationError('invalid family optimizer options')
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
    p.tol ?? 1e-12,
    get('polishBatchSize', 'polish_batch_size', 0),
    p.hierarchical == null ? 0 : typeof p.hierarchical === 'boolean' ? Number(p.hierarchical) : NaN,
    get('localRefineInterval', 'local_refine_interval', 0),
    get('localRefineCount', 'local_refine_count', 0),
    Number(scaled),
    Number(method === 'lm')
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
  if (ArrayBuffer.isView(X)) X = { rows: X.length / cols, cols, data: X }
  const m = normalizeX(X)
  if (m.cols !== cols || !m.data.every(v => Number.isFinite(Math.fround(v))))
    throw new ValidationError('family matrix shape or values invalid')
  return m
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
    if (this.params.backend === 'polygrad') return this._fitPolygrad(X, target, options)
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
    this.artifactVersion = this.params.hierarchical ? 3 : 2
    return this
  }
  async predictPolygrad(X, opts = {}) {
    const { predict } = require('./family-refine.js')
    const m = matrix(X, this.nFeatures)
    const heads = this.classes.length > 2 ? this.classes.length : 1
    const values = []
    // Copy formulas before the first await; no borrowed C state crosses yields.
    const candidates = Array.from({ length: heads }, (_, h) => this._candidate(h, -1))
    for (const c of candidates)
      values.push(
        await predict(
          c,
          m,
          opts.polygradRuntime ||
            opts.polygrad ||
            this.params.polygradRuntime ||
            this.params.polygrad
        )
      )
    if (!this.classes.length) return values[0]
    return Float64Array.from({ length: m.rows }, (_, row) => {
      let best = 0
      for (let h = 1; h < heads; h++) if (values[h][row] > values[best][row]) best = h
      return this.classes[heads === 1 ? Number(values[0][row] >= 0) : best]
    })
  }
  async refinePolygrad(X, y, opts = {}) {
    const { refine } = require('./family-refine.js')
    const m = matrix(X, this.nFeatures),
      head = opts.index ?? 0
    const heads = this.classes.length > 2 ? this.classes.length : 1
    if (!Number.isInteger(head) || head < 0 || head >= heads)
      throw new ValidationError('invalid family head index')
    if (y.length !== m.rows) throw new ValidationError('family target length mismatch')
    const labels = new Map(this.classes.map((v, i) => [v, i]))
    const target = Float64Array.from(y, v => (this.classes.length ? (labels.get(v) ?? NaN) : v))
    const fraction = this.params.validationFraction ?? this.params.validation_fraction ?? 0
    const seed = this.params.seed ?? 42
    const handle = this.handle
    const data = withArrays([target, new Float64Array(2 * m.rows + 1)], (w, yp, out) => {
      if (w._sym_family_refine_data(handle, head, yp, m.rows, fraction, seed, out, 2 * m.rows + 1))
        throw error(w)
      return w.HEAPF64.slice(out / 8, out / 8 + 2 * m.rows + 1)
    })
    const trainX = [],
      trainY = []
    for (let r = 0; r < m.rows; r++)
      if (data[m.rows + r]) {
        for (let j = 0; j < m.cols; j++) trainX.push(m.data[r * m.cols + j])
        trainY.push(data[r])
      }
    const { updates, history } = await refine(
      this._candidate(head, -1),
      { rows: trainY.length, cols: m.cols, data: trainX },
      trainY,
      { ...this.params, ...opts }
    )
    if (!this.handle || this.handle !== handle)
      throw new ValidationError('family model replaced or disposed during refinement')
    return withArrays([updates, m.data, target, new Float64Array(6)], (w, up, xp, yp, out) => {
      const rc = w._sym_family_refine_accept(
        handle,
        head,
        up,
        updates.length,
        xp,
        m.rows,
        m.cols,
        yp,
        fraction,
        seed,
        opts.tolerance ?? 1e-10,
        out
      )
      if (rc < 0) throw error(w)
      const v = w.HEAPF64.slice(out / 8, out / 8 + 6)
      return {
        status: 'ok',
        committed: !!rc,
        head,
        history,
        beforeLoss: v[0],
        beforeValidLoss: v[1],
        beforeObjective: v[2],
        afterLoss: v[3],
        afterValidLoss: v[4],
        afterObjective: v[5]
      }
    })
  }
  async _fitPolygrad(X, target, options) {
    const { FamilyScorer } = require('./family-scorer.js')
    const w = getWasm()
    const capacity = this.params.batchSize ?? this.params.batch_size ?? Math.min(options[0], 512)
    if (!Number.isInteger(capacity) || capacity < 1 || capacity > 512)
      throw new ValidationError('family batchSize must be an integer in [1,512]')
    const search = withArrays([X.data, target, options], (w, xp, yp, pp) =>
      w._wl_sym_family_search_new(
        xp,
        X.rows,
        X.cols,
        yp,
        this.classes.length ? 1 : 0,
        this.classes.length,
        pp,
        options.length,
        capacity,
        Math.min(X.rows, 4096)
      )
    )
    if (!search) throw error(w)
    const terms = options[2]
    const descriptors = new Float64Array(capacity * terms * 5)
    let scorer,
      head = -1,
      actual,
      identity
    const solve = (factors, stats, width) =>
      withArrays(
        [
          Float64Array.from(factors.subarray(0, actual * width * width)),
          Float64Array.from(stats.subarray(0, actual * terms * 3)),
          new Float64Array(actual * (terms + 1))
        ],
        (w, fp, sp, out) => {
          if (
            w._sym_family_search_solve(
              search,
              identity,
              fp,
              actual * width * width,
              sp,
              actual * terms * 3,
              width,
              out,
              actual * (terms + 1)
            )
          )
            throw error(w)
          const values = new Float64Array(capacity * (terms + 1))
          values.set(w.HEAPF64.subarray(out / 8, out / 8 + actual * (terms + 1)))
          for (let c = actual; c < capacity; c++)
            values.set(values.subarray(0, terms + 1), c * (terms + 1))
          return values
        }
      )
    try {
      for (;;) {
        const rc = w._wl_sym_family_search_propose(search)
        if (rc < 0) throw error(w)
        if (!rc) break
        identity = w._wl_sym_family_batch_id(search)
        const shape = withArrays([new Float64Array(4)], (w, ptr) => {
          if (w._wl_sym_family_batch_shape(search, ptr, 8)) throw error(w)
          return w.HEAP32.slice(ptr / 4, ptr / 4 + 8)
        })
        actual = shape[1]
        withArrays([new Float64Array(actual * terms * 5)], (w, ptr) => {
          if (w._wl_sym_family_batch_descriptors(search, ptr, actual * terms * 5)) throw error(w)
          descriptors.set(w.HEAPF64.subarray(ptr / 8, ptr / 8 + actual * terms * 5))
        })
        for (let c = actual; c < capacity; c++)
          descriptors.set(descriptors.subarray(0, terms * 5), c * terms * 5)
        if (head !== shape[6]) {
          if (scorer) scorer.dispose()
          head = shape[6]
          const data = withArrays([new Float64Array(2 * X.rows + 1)], (w, ptr) => {
            if (w._sym_family_search_data(search, ptr, 2 * X.rows + 1)) throw error(w)
            return w.HEAPF64.slice(ptr / 8, ptr / 8 + 2 * X.rows + 1)
          })
          scorer = await FamilyScorer.create(
            X,
            data.slice(0, X.rows),
            data.slice(X.rows, 2 * X.rows),
            data[2 * X.rows],
            descriptors,
            capacity,
            terms,
            solve,
            this.params
          )
        }
        const needResiduals = w._sym_family_search_residual_count(search) > 0
        const { coefficients, losses, residuals } = await scorer.score(descriptors, needResiduals)
        const width = terms + 3 + (needResiduals ? X.rows : 0)
        const packed = new Float64Array(actual * width)
        for (let c = 0; c < actual; c++) {
          packed.set(coefficients.subarray(c * (terms + 1), (c + 1) * (terms + 1)), c * width)
          packed[c * width + terms + 1] = losses[2 * c]
          packed[c * width + terms + 2] = losses[2 * c + 1]
          if (needResiduals)
            packed.set(residuals.subarray(c * X.rows, (c + 1) * X.rows), c * width + terms + 3)
        }
        withArrays([packed], (w, ptr) => {
          if (w._sym_family_search_accept_results(search, identity, ptr, packed.length))
            throw error(w)
        })
      }
      const model = w._sym_family_search_finish(search)
      if (!model) throw error(w)
      this.dispose()
      this.handle = model
      this.nFeatures = X.cols
      this.terms = terms
      this.artifactVersion = this.params.hierarchical ? 3 : 2
      return this
    } finally {
      if (scorer) scorer.dispose()
      w._sym_family_search_free(search)
    }
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
    if (opts.backend === 'polygrad') return this.predictPolygrad(X, opts)
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
      kind: this.artifactVersion === 3 ? 'sym.family.formula@3' : 'sym.pg-family.formula@1',
      ...(this.artifactVersion === 3
        ? { nFeatures: this.nFeatures, sourceEncoding: 'inputs-then-earlier-terms' }
        : {}),
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
      m.artifactVersion = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(
        12,
        true
      )
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

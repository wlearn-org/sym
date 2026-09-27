'use strict'

const { getWasm, loadSym } = require('./wasm.js')
const { CFamilyEngine } = require('./family-c.js')
const { resolveStrategy, familySearchSpace } = require('./strategy.js')
const { FormulaVerifier } = require('./verifier.js')
const { loadPolygrad, evaluateFormulaPolygrad, refineFormulaPolygrad } = require('./polygrad.js')
const { PgFamilyRegressorEngine, PgFamilyClassifierEngine } = require('./engine-pg-family.js')
const {
  encodeBundle,
  decodeBundle,
  register,
  ValidationError,
  normalizeX: normalizeCoreX,
  NotFittedError,
  DisposedError
} = require('@wlearn/core')

const TYPE_ID_REGRESSOR = 'wlearn.sym.regressor@1'
const TYPE_ID_CLASSIFIER = 'wlearn.sym.classifier@1'
const TYPE_ID_TRANSFORMER = 'wlearn.sym.transformer@1'

const TASK = { regression: 0, classification: 1, transformer: 2 }
const LOSS = { mse: 0, mae: 1, huber: 2, logloss: 3 }
const OPSET = { basic: 0, smooth: 1, full: 2 }
const FINAL_SELECTOR = { objective: 0, loss: 1, validloss: 1, validationloss: 1, score: 2 }
const ENGINE_C = 'c'
const ENGINE_PG_FAMILY = 'pg-family'
const FAMILY_MEDIA = 'application/vnd.wlearn.sym.family'
const PG_FAMILY_MEDIA = 'application/vnd.wlearn.sym.pg-family+json'

function getNodeFs() {
  try {
    return require('node:fs')
  } catch (_) {
    return null
  }
}

function readBytesOrPath(bytesOrPath) {
  if (typeof bytesOrPath === 'string') {
    const fs = getNodeFs()
    if (!fs || typeof fs.readFileSync !== 'function') {
      throw new Error('loading from a filesystem path is only available in Node')
    }
    return fs.readFileSync(bytesOrPath)
  }
  if (bytesOrPath instanceof Uint8Array) return bytesOrPath
  if (bytesOrPath instanceof ArrayBuffer) return new Uint8Array(bytesOrPath)
  return bytesOrPath
}

function writeBytesMaybe(bytes, path) {
  if (path == null) return bytes
  const fs = getNodeFs()
  if (!fs || typeof fs.writeFileSync !== 'function') {
    throw new Error('saving to a filesystem path is only available in Node')
  }
  fs.writeFileSync(path, bytes)
  return bytes
}

function getLastError() {
  return getWasm().ccall('wl_sym_get_last_error', 'string', [], [])
}

function resolveEnum(map, value, fallback) {
  if (typeof value === 'number') return value
  if (typeof value === 'string' && Object.prototype.hasOwnProperty.call(map, value.toLowerCase())) {
    return map[value.toLowerCase()]
  }
  return fallback
}

function normalizeX(X, nFeatures = 0) {
  if (X && typeof X === 'object' && X.data != null) {
    const matrix = normalizeCoreX(X)
    if ((nFeatures && matrix.cols !== nFeatures) || !matrix.data.every(Number.isFinite))
      throw new ValidationError('invalid matrix values or feature count')
    return matrix
  }
  if (X instanceof Float64Array) {
    if (!nFeatures) throw new Error('nFeatures is required for flat Float64Array input')
    if (X.length % nFeatures !== 0)
      throw new Error(`flat X length ${X.length} is not divisible by nFeatures ${nFeatures}`)
    return { rows: X.length / nFeatures, cols: nFeatures, data: X }
  }
  if (X instanceof Float32Array) {
    if (!nFeatures) throw new Error('nFeatures is required for flat Float32Array input')
    if (X.length % nFeatures !== 0)
      throw new Error(`flat X length ${X.length} is not divisible by nFeatures ${nFeatures}`)
    return { rows: X.length / nFeatures, cols: nFeatures, data: new Float64Array(X) }
  }
  if (!Array.isArray(X))
    throw new Error('X must be an array of rows or a Float64Array/Float32Array')
  const rows = X.length
  const cols = rows > 0 && Array.isArray(X[0]) ? X[0].length : 1
  const data = new Float64Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    const row = Array.isArray(X[i]) ? X[i] : [X[i]]
    if (row.length !== cols)
      throw new Error(`X row ${i} has ${row.length} columns, expected ${cols}`)
    for (let j = 0; j < cols; j++) {
      const v = Number(row[j])
      if (!Number.isFinite(v)) throw new Error(`X[${i},${j}] must be finite`)
      data[i * cols + j] = v
    }
  }
  return { rows, cols, data }
}

function normalizeY(y, rows) {
  const arr = y instanceof Float64Array ? y : new Float64Array(y)
  if (arr.length !== rows)
    throw new Error(`y length (${arr.length}) does not match X rows (${rows})`)
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) throw new Error(`y[${i}] must be finite`)
  }
  return arr
}

function meanSquaredError(y, pred) {
  if (y.length !== pred.length)
    throw new Error(`prediction length ${pred.length} does not match y length ${y.length}`)
  let loss = 0
  for (let i = 0; i < y.length; i++) {
    const e = pred[i] - y[i]
    loss += e * e
  }
  return loss / Math.max(1, y.length)
}

function uniqueSorted(y) {
  return [...new Set(Array.from(y, Number))].sort((a, b) => a - b)
}

function mapClasses(y, classes) {
  const index = new Map(classes.map((c, i) => [c, i]))
  const out = new Float64Array(y.length)
  for (let i = 0; i < y.length; i++) {
    const key = Number(y[i])
    if (!index.has(key)) throw new Error(`unknown class label: ${key}`)
    out[i] = index.get(key)
  }
  return out
}

function writeArrayToWasm(wasm, data, bytesPerElement) {
  const ptr = wasm._malloc(data.length * bytesPerElement)
  if (bytesPerElement === 8) wasm.HEAPF64.set(data, ptr / 8)
  else wasm.HEAPU8.set(data, ptr)
  return ptr
}

function readCStringBuffer(wasm, callFn) {
  const ptrPtr = wasm._malloc(4)
  const lenPtr = wasm._malloc(4)
  try {
    const ret = callFn(ptrPtr, lenPtr)
    if (ret !== 0) throw new Error(getLastError())
    const ptr = wasm.HEAP32[ptrPtr / 4]
    const len = wasm.HEAP32[lenPtr / 4]
    const bytes = new Uint8Array(wasm.HEAPU8.buffer, ptr, len).slice()
    wasm._wl_sym_free_buffer(ptr)
    return bytes
  } finally {
    wasm._free(ptrPtr)
    wasm._free(lenPtr)
  }
}

const textDecoder = new TextDecoder()
const textEncoder = new TextEncoder()

function parseJsonBytes(bytes) {
  return JSON.parse(textDecoder.decode(bytes))
}

class BaseSymModel {
  constructor(params = {}, taskName = 'regression') {
    this._handle = null
    this._params = { ...params }
    this._taskName = taskName
    this._nFeatures = 0
    this._nOutputs = 0
    this._nClasses = 0
    this._classes = []
    this._fitted = false
    this._disposed = false
    this._polygrad = null
    this._ownsPolygrad = false
    this._choice = resolveStrategy(params, taskName)
    this._engineName = this._choice.legacy ? ENGINE_PG_FAMILY : ENGINE_C
    this._familyEngine = null
  }

  static async _create(cls, params) {
    const model = new cls(params)
    if (model._engineName !== ENGINE_PG_FAMILY) {
      await loadSym(params && params.wasm)
      if (params && params.polygrad && model._choice.strategy === 'tree')
        await model._getPolygrad(params.polygrad)
    }
    return model
  }

  get isFitted() {
    return this._fitted && !this._disposed
  }
  get nFeatures() {
    return this._nFeatures
  }
  get nOutputs() {
    return this._nOutputs
  }
  get nClasses() {
    return this._nClasses
  }
  get classes() {
    return this._classes.slice()
  }

  fit(X, y) {
    if (this._disposed) throw new DisposedError(`${this.constructor.name} has been disposed.`)
    const { rows, cols, data } = normalizeX(X)
    const yArr = normalizeY(y, rows)
    if (this._choice.strategy === 'family' && !this._choice.legacy) {
      const classes =
        this._taskName === 'classification'
          ? this._params.classes
            ? Array.from(this._params.classes, Number)
            : uniqueSorted(yArr)
          : []
      if (
        this._taskName === 'classification' &&
        (classes.length < 2 ||
          classes.some(c => !Number.isFinite(c)) ||
          new Set(classes).size !== classes.length)
      )
        throw new ValidationError('classification requires unique finite classes')
      const engine = new CFamilyEngine(this._params, classes)
      try {
        engine.fit({ rows, cols, data }, yArr)
      } catch (err) {
        engine.dispose()
        throw err
      }
      this._freeHandle()
      this._familyEngine = engine
      this._classes = classes
      this._nClasses = classes.length
      this._nOutputs = classes.length > 2 ? classes.length : 1
      this._nFeatures = cols
      this._fitted = true
      return this
    }
    this._freeHandle()
    if (this._engineName === ENGINE_PG_FAMILY) {
      return this._fitPgFamily({ rows, cols, data }, yArr)
    }
    const wasm = getWasm()

    let task = TASK[this._taskName]
    let nClasses = 0
    let topK = Number(this._params.topK ?? this._params.top_k ?? 8)
    let fitY = yArr

    if (this._taskName === 'classification') {
      this._classes = this._params.classes
        ? Array.from(this._params.classes, Number)
        : uniqueSorted(yArr)
      if (this._classes.length < 2) throw new Error('classification requires at least two classes')
      nClasses = this._classes.length
      fitY = mapClasses(yArr, this._classes)
    }

    const xPtr = writeArrayToWasm(wasm, data, 8)
    const yPtr = writeArrayToWasm(wasm, fitY, 8)
    const handle = wasm._wl_sym_fit(
      xPtr,
      rows,
      cols,
      yPtr,
      task,
      nClasses,
      topK,
      this._params.population ?? 256,
      this._params.generations ?? 120,
      this._params.maxNodes ?? this._params.max_nodes ?? 31,
      this._params.maxDepth ?? this._params.max_depth ?? 6,
      this._params.frontierSize ?? this._params.frontier_size ?? 16,
      this._params.tournamentSize ?? this._params.tournament_size ?? 4,
      this._params.eliteCount ?? this._params.elite_count ?? 4,
      resolveEnum(
        LOSS,
        this._params.loss,
        this._taskName === 'classification' ? LOSS.logloss : LOSS.mse
      ),
      resolveEnum(OPSET, this._params.operatorSet ?? this._params.operator_set, OPSET.full),
      this._params.earlyStopRounds ?? this._params.early_stop_rounds ?? 30,
      this._params.seed ?? 42,
      this._params.validationFraction ?? this._params.validation_fraction ?? 0,
      this._params.complexityPenalty ?? this._params.complexity_penalty ?? 0.001,
      this._params.mutationRate ?? this._params.mutation_rate ?? 0.35,
      this._params.crossoverRate ?? this._params.crossover_rate ?? 0.55,
      this._params.constantRate ?? this._params.constant_rate ?? 0.2,
      this._params.constMin ?? this._params.const_min ?? -4.0,
      this._params.constMax ?? this._params.const_max ?? 4.0,
      this._params.huberDelta ?? this._params.huber_delta ?? 1.0,
      this._params.tol ?? 1e-12,
      this._params.islands ?? 1,
      this._params.migrationInterval ?? this._params.migration_interval ?? 0,
      this._params.migrationCount ?? this._params.migration_count ?? 0,
      this._params.warmupGenerations ?? this._params.warmup_generations ?? 0,
      this._params.warmupMinNodes ?? this._params.warmup_min_nodes ?? 0,
      this._params.broodSize ?? this._params.brood_size ?? 1,
      this._params.rowSampleSize ?? this._params.row_sample_size ?? 0,
      this._params.localRefineInterval ?? this._params.local_refine_interval ?? 0,
      this._params.localRefineCount ?? this._params.local_refine_count ?? 0,
      this._params.complexityHofSize ?? this._params.complexity_hof_size ?? 0,
      resolveEnum(
        FINAL_SELECTOR,
        this._params.finalSelector ?? this._params.final_selector,
        FINAL_SELECTOR.objective
      ),
      this._params.complexityBucketWidth ?? this._params.complexity_bucket_width ?? 2.0
    )
    wasm._free(xPtr)
    wasm._free(yPtr)

    if (!handle) throw new Error(`Symbolic fit failed: ${getLastError()}`)

    this._handle = handle
    this._fitted = true
    this._nFeatures = wasm._wl_sym_get_n_features(handle)
    this._nOutputs = wasm._wl_sym_get_n_outputs(handle)
    this._nClasses = wasm._wl_sym_get_n_classes(handle)
    return this
  }

  async _fitPgFamily(matrix, yArr) {
    if (this._taskName === 'transformer') {
      throw new Error(
        'engine="pg-family" currently supports SymbolicRegressor and SymbolicClassifier; use engine="c" for FormulaTransformer'
      )
    }
    const params = { ...this._params }
    let engine
    if (this._taskName === 'classification') {
      this._classes = params.classes ? Array.from(params.classes, Number) : uniqueSorted(yArr)
      params.classes = this._classes
      engine = await PgFamilyClassifierEngine.create(params)
      await engine.fit(matrix, yArr)
      this._nClasses = this._classes.length
      this._nOutputs = this._nClasses === 2 ? 1 : this._nClasses
    } else {
      engine = await PgFamilyRegressorEngine.create(params)
      await engine.fit(matrix, yArr)
      this._nOutputs = 1
      this._nClasses = 0
      this._classes = []
    }
    this._familyEngine = engine
    this._fitted = true
    this._nFeatures = matrix.cols
    return this
  }

  predict(X) {
    this._ensureFitted()
    if (this._familyEngine) return this._familyEngine.predict(X)
    const { rows, cols, data } = normalizeX(X, this._nFeatures)
    if (cols !== this._nFeatures)
      throw new Error(`X has ${cols} columns, expected ${this._nFeatures}`)
    const wasm = getWasm()
    const xPtr = writeArrayToWasm(wasm, data, 8)
    const outLen = this._taskName === 'transformer' ? rows * this._nOutputs : rows
    const outPtr = wasm._malloc(outLen * 8)
    try {
      const ret = wasm._wl_sym_predict(this._handle, xPtr, rows, cols, outPtr)
      if (ret !== 0) throw new Error(`Symbolic predict failed: ${getLastError()}`)
      const out = new Float64Array(wasm.HEAPF64.buffer, outPtr, outLen).slice()
      if (this._taskName === 'classification' && this._classes.length) {
        const mapped = new Float64Array(out.length)
        for (let i = 0; i < out.length; i++) mapped[i] = this._classes[out[i] | 0]
        return mapped
      }
      return out
    } finally {
      wasm._free(xPtr)
      wasm._free(outPtr)
    }
  }

  decisionFunction(X) {
    this._ensureFitted()
    if (this._familyEngine)
      return this._familyEngine.decisionFunction
        ? this._familyEngine.decisionFunction(X)
        : this._familyEngine.predict(X)
    const { rows, cols, data } = normalizeX(X, this._nFeatures)
    if (cols !== this._nFeatures)
      throw new Error(`X has ${cols} columns, expected ${this._nFeatures}`)
    const wasm = getWasm()
    const xPtr = writeArrayToWasm(wasm, data, 8)
    const outPtr = wasm._malloc(rows * this._nOutputs * 8)
    try {
      const ret = wasm._wl_sym_predict_raw(this._handle, xPtr, rows, cols, outPtr)
      if (ret !== 0) throw new Error(`Symbolic raw prediction failed: ${getLastError()}`)
      return new Float64Array(wasm.HEAPF64.buffer, outPtr, rows * this._nOutputs).slice()
    } finally {
      wasm._free(xPtr)
      wasm._free(outPtr)
    }
  }

  score(X, y) {
    this._ensureFitted()
    if (this._familyEngine) return this._familyEngine.score(X, y)
    const { rows, cols, data } = normalizeX(X, this._nFeatures)
    const yArr = normalizeY(y, rows)
    const fitY =
      this._taskName === 'classification' && this._classes.length
        ? mapClasses(yArr, this._classes)
        : yArr
    const wasm = getWasm()
    const xPtr = writeArrayToWasm(wasm, data, 8)
    const yPtr = writeArrayToWasm(wasm, fitY, 8)
    try {
      const result = wasm._wl_sym_score(this._handle, xPtr, rows, cols, yPtr)
      if (!Number.isFinite(result)) throw new Error(`Symbolic score failed: ${getLastError()}`)
      return result
    } finally {
      wasm._free(xPtr)
      wasm._free(yPtr)
    }
  }

  formula(opts = {}) {
    this._ensureFitted()
    if (this._familyEngine) return this._familyEngine.formula(opts)
    const format = opts.format || 'json'
    const index = opts.index ?? 0
    const bytes = this._formulaBytes(index, format)
    const text = textDecoder.decode(bytes)
    return format === 'text' ? text : JSON.parse(text)
  }

  frontier(opts = {}) {
    this._ensureFitted()
    if (this._familyEngine) return this._familyEngine.frontier(opts)
    const wasm = getWasm()
    const nFormulas = wasm._wl_sym_get_n_formulas(this._handle)
    const n = wasm._wl_sym_get_frontier_size(this._handle)
    const out = []
    for (let i = 0; i < n; i++) {
      const index = nFormulas + i
      const json = this.formula({ ...opts, index, format: 'json' })
      json.index = i
      json.text = this.formula({ index, format: 'text' })
      out.push(json)
    }
    return out
  }

  verify(checks = {}) {
    return FormulaVerifier.verify(this.formula({ format: 'json' }), {
      nFeatures: this._nFeatures,
      ...checks
    })
  }

  async predictPolygrad(X, opts = {}) {
    this._ensureFitted()
    if (this._familyEngine) {
      if (this._taskName !== 'regression')
        throw new Error('predictPolygrad for pg-family currently supports regression only')
      return this._familyEngine.predict(X, { backend: 'polygrad', ...opts })
    }
    const runtime = await this._getPolygrad(opts.polygrad || this._params.polygrad)
    const outputs = []
    const formulas =
      this._taskName === 'transformer'
        ? Array.from({ length: this._nOutputs }, (_, index) =>
            this.formula({ index, format: 'json' })
          )
        : [this.formula({ index: 0, format: 'json' })]
    for (const formula of formulas) {
      outputs.push(await evaluateFormulaPolygrad(runtime, formula, X, this._nFeatures))
    }
    if (outputs.length === 1) return outputs[0]
    const rows = outputs[0].length
    const out = new Float64Array(rows * outputs.length)
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < outputs.length; c++) out[r * outputs.length + c] = outputs[c][r]
    }
    return out
  }

  async _getPolygrad(options) {
    if (!this._polygrad) {
      options = await options
      this._polygrad = await loadPolygrad(options)
      this._ownsPolygrad = !(options && options.Tensor)
    }
    return this._polygrad
  }

  async refinePolygrad(X, y, opts = {}) {
    this._ensureFitted()
    if (this._familyEngine) {
      return {
        status: 'unsupported',
        reason:
          'post-fit Polygrad refinement is available for tree formulas only; family models retain their fitted readout',
        committed: false
      }
    }
    if (this._taskName !== 'regression') {
      throw new Error('refinePolygrad currently supports regression formulas only')
    }
    const index = opts.index ?? 0
    if (index !== 0)
      throw new Error('Refinement supports the selected regression formula (index 0)')
    const { rows, cols } = normalizeX(X, this._nFeatures)
    if (cols !== this._nFeatures)
      throw new Error(`X has ${cols} columns, expected ${this._nFeatures}`)
    const yArr = normalizeY(y, rows)
    const beforePred = this.predict(X)
    const beforeLoss = meanSquaredError(yArr, beforePred)
    const beforeScore = this.score(X, yArr)
    const formula = this.formula({ index, format: 'json' })
    const runtime = await this._getPolygrad(opts.polygrad || this._params.polygrad)
    const report = await refineFormulaPolygrad(runtime, formula, X, yArr, this._nFeatures, opts)
    if (report.status !== 'ok') {
      return {
        ...report,
        committed: false,
        beforeLoss,
        beforeScore
      }
    }

    const tolerance = Number(opts.tolerance ?? 1e-10)
    if (!Number.isFinite(tolerance) || tolerance < 0)
      throw new Error('tolerance must be finite and nonnegative')
    // Validate constants with the authoritative C evaluator before committing.
    const candidate = await this.constructor.load(this.save())
    try {
      for (const c of report.committedConstants)
        candidate._setFormulaConstant(index, c.nodeIndex, c.value)
      const afterLoss = meanSquaredError(yArr, candidate.predict(X))
      const accepted = Number.isFinite(afterLoss) && afterLoss <= beforeLoss + tolerance
      if (accepted) {
        // Keep search-time metrics; these rows are not the original held-out split.
        const previous = this._handle
        this._handle = candidate._handle
        candidate._handle = previous
      }
      return {
        ...report,
        loss: afterLoss,
        status: accepted
          ? afterLoss + tolerance < beforeLoss
            ? 'improved'
            : 'unchanged'
          : 'rejected',
        committed: accepted,
        beforeLoss,
        beforeScore,
        afterLoss,
        afterScore: this.score(X, yArr)
      }
    } finally {
      candidate.dispose()
    }
  }

  save(path) {
    this._ensureFitted()
    if (this._familyEngine) {
      const binary = this._familyEngine.artifactVersion === 2
      const data = binary
        ? this._familyEngine.toBytes()
        : textEncoder.encode(JSON.stringify(this._familyEngine.toState()))
      const bundle = encodeBundle(
        {
          typeId: binary ? this.constructor.typeId.replace('@1', '@2') : this.constructor.typeId,
          params: this.getParams(),
          metadata: {
            ...(this._choice.legacy ? { engine: ENGINE_PG_FAMILY } : {}),
            strategy: 'family',
            backend: this._choice.backend,
            nFeatures: this._nFeatures,
            nOutputs: this._nOutputs,
            nClasses: this._nClasses,
            classes: this._classes,
            task: this._taskName
          }
        },
        [
          {
            id: 'model',
            mediaType: binary ? FAMILY_MEDIA : PG_FAMILY_MEDIA,
            data
          }
        ]
      )
      return writeBytesMaybe(bundle, path)
    }
    const raw = this._saveRaw()
    const bundle = encodeBundle(
      {
        typeId: this.constructor.typeId,
        params: this.getParams(),
        metadata: {
          nFeatures: this._nFeatures,
          nOutputs: this._nOutputs,
          nClasses: this._nClasses,
          classes: this._classes,
          task: this._taskName
        }
      },
      [
        {
          id: 'model',
          mediaType: 'application/vnd.wlearn.sym.raw',
          data: raw
        }
      ]
    )
    return writeBytesMaybe(bundle, path)
  }

  static async load(bytesOrPath) {
    const { manifest, toc, blobs } = decodeBundle(readBytesOrPath(bytesOrPath))
    return this._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    const binary =
      manifest.typeId === this.typeId.replace('@1', '@2') && this.typeId !== TYPE_ID_TRANSFORMER
    if (manifest.typeId !== this.typeId && !binary) {
      throw new Error(`Unsupported sym bundle typeId: ${manifest.typeId}`)
    }
    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)
    if (binary) {
      if (entry.mediaType !== FAMILY_MEDIA || manifest.params?.strategy !== 'family')
        throw new ValidationError('invalid family artifact type or strategy')
      await loadSym()
      const model = new this(manifest.params)
      try {
        const classes = manifest.metadata?.classes
        if (!Array.isArray(classes) || (this.typeId === TYPE_ID_CLASSIFIER) !== classes.length > 0)
          throw new ValidationError('family payload task mismatch')
        model._familyEngine = CFamilyEngine.fromBytes(raw, model._params, classes)
        model._nFeatures = model._familyEngine.nFeatures
        model._classes = [...classes]
        model._nClasses = classes.length
        model._nOutputs = classes.length > 2 ? classes.length : 1
        if (
          manifest.metadata.nFeatures !== model._nFeatures ||
          manifest.metadata.nClasses !== model._nClasses ||
          manifest.metadata.nOutputs !== model._nOutputs
        )
          throw new ValidationError('family artifact dimensions mismatch')
        model._fitted = true
        return model
      } catch (err) {
        model.dispose()
        throw err
      }
    }
    if (entry.mediaType === FAMILY_MEDIA)
      throw new ValidationError('family binary requires an @2 typeId')
    if (
      entry.mediaType === PG_FAMILY_MEDIA ||
      (manifest.metadata && manifest.metadata.engine === ENGINE_PG_FAMILY)
    ) {
      if (manifest.typeId === TYPE_ID_TRANSFORMER) {
        throw new Error(
          'pg-family bundles are supported for SymbolicRegressor and SymbolicClassifier only'
        )
      }
      const state = parseJsonBytes(raw)
      if (manifest.params?.strategy === 'family' && manifest.params?.backend === 'c') {
        const expected = manifest.typeId === TYPE_ID_CLASSIFIER ? 'classifier' : 'regressor'
        if (state?.kind !== `wlearn.sym.pg-family.${expected}@1`)
          throw new ValidationError('family payload task does not match bundle type')
        await loadSym()
        const model = new this(manifest.params)
        model._familyEngine = CFamilyEngine.fromState(state, manifest.params)
        model._fitted = true
        model._nFeatures = model._familyEngine.nFeatures
        model._classes = model._familyEngine.classes
        model._nClasses = model._classes.length
        model._nOutputs = model._nClasses > 2 ? model._nClasses : 1
        return model
      }
      const model = new this({ ...(manifest.params || {}), engine: ENGINE_PG_FAMILY })
      model._engineName = ENGINE_PG_FAMILY
      if (manifest.typeId === TYPE_ID_CLASSIFIER) {
        model._familyEngine = PgFamilyClassifierEngine.fromStateSync(state)
      } else {
        model._familyEngine = PgFamilyRegressorEngine.fromStateSync(state)
      }
      model._fitted = true
      model._nFeatures = Number(
        (manifest.metadata && manifest.metadata.nFeatures) || model._familyEngine.nFeatures || 0
      )
      model._nOutputs = Number(
        (manifest.metadata && manifest.metadata.nOutputs) ||
          (manifest.typeId === TYPE_ID_CLASSIFIER
            ? model._familyEngine.classes.length === 2
              ? 1
              : model._familyEngine.classes.length
            : 1)
      )
      model._nClasses = Number((manifest.metadata && manifest.metadata.nClasses) || 0)
      model._classes =
        manifest.metadata && manifest.metadata.classes
          ? [...manifest.metadata.classes]
          : model._familyEngine.classes || []
      return model
    }
    await loadSym()
    const wasm = getWasm()
    const ptr = wasm._malloc(raw.length)
    wasm.HEAPU8.set(raw, ptr)
    const handle = wasm._wl_sym_load(ptr, raw.length)
    wasm._free(ptr)
    if (!handle) throw new Error(`Symbolic load failed: ${getLastError()}`)
    const model = new this(manifest.params || {})
    model._handle = handle
    model._fitted = true
    model._nFeatures = wasm._wl_sym_get_n_features(handle)
    model._nOutputs = wasm._wl_sym_get_n_outputs(handle)
    model._nClasses = wasm._wl_sym_get_n_classes(handle)
    model._classes =
      manifest.metadata && manifest.metadata.classes ? [...manifest.metadata.classes] : []
    return model
  }

  dispose() {
    if (this._disposed) return
    this._disposed = true
    this._freeHandle()
    this._freeFamilyEngine()
    if (this._ownsPolygrad && this._polygrad) this._polygrad.dispose()
    this._polygrad = null
  }

  getParams() {
    const { wasm, polygrad, polygradRuntime, ...rest } = this._params
    return { ...rest, strategy: this._choice.strategy, backend: this._choice.backend }
  }

  setParams(params = {}) {
    const merged = { ...this._params, ...params }
    const choice = resolveStrategy(merged, this._taskName)
    if (
      choice.strategy !== this._choice.strategy ||
      choice.backend !== this._choice.backend ||
      choice.legacy !== this._choice.legacy
    )
      this._freeHandle()
    this._params = merged
    this._choice = choice
    this._engineName = choice.legacy ? ENGINE_PG_FAMILY : ENGINE_C
    return this
  }

  get capabilities() {
    return {
      classifier: this._taskName === 'classification',
      regressor: this._taskName === 'regression',
      transformer: this._taskName === 'transformer',
      predictProba: this._taskName === 'classification',
      decisionFunction: this._taskName === 'classification' || this._taskName === 'regression',
      sampleWeight: false,
      csr: false,
      earlyStopping: true,
      featureImportances: false,
      formulas: true,
      polygradExecution: this._choice.strategy === 'tree' || this._choice.legacy,
      polygradRefinement: this._taskName === 'regression' && this._choice.strategy === 'tree',
      engines: this._taskName === 'transformer' ? [ENGINE_C] : [ENGINE_C, ENGINE_PG_FAMILY],
      strategies: this._taskName === 'transformer' ? ['tree'] : ['tree', 'family'],
      backends: ['c'],
      activeStrategy: this._choice.strategy,
      activeBackend: this._choice.backend,
      legacySearch: this._choice.legacy,
      activeEngine: this._engineName
    }
  }

  _formulaBytes(index, format) {
    const wasm = getWasm()
    return readCStringBuffer(wasm, (ptrPtr, lenPtr) => {
      if (format === 'text') return wasm._wl_sym_formula_text(this._handle, index, ptrPtr, lenPtr)
      return wasm._wl_sym_formula_json(this._handle, index, ptrPtr, lenPtr)
    })
  }

  _saveRaw() {
    const wasm = getWasm()
    return readCStringBuffer(wasm, (ptrPtr, lenPtr) =>
      wasm._wl_sym_save(this._handle, ptrPtr, lenPtr)
    )
  }

  _setFormulaConstant(index, nodeIndex, value) {
    const wasm = getWasm()
    const ret = wasm._wl_sym_set_formula_constant(this._handle, index, nodeIndex, value)
    if (ret !== 0) throw new Error(`Symbolic constant update failed: ${getLastError()}`)
  }

  _setFormulaMetrics(index, trainLoss, validLoss, objective) {
    const wasm = getWasm()
    const ret = wasm._wl_sym_set_formula_metrics(
      this._handle,
      index,
      trainLoss,
      validLoss,
      objective
    )
    if (ret !== 0) throw new Error(`Symbolic metric update failed: ${getLastError()}`)
  }

  _ensureFitted() {
    if (this._disposed) throw new DisposedError(`${this.constructor.name} has been disposed.`)
    if (!this._fitted || (!this._handle && !this._familyEngine))
      throw new NotFittedError(`${this.constructor.name} is not fitted. Call fit() first.`)
  }

  _freeHandle() {
    if (this._handle) {
      getWasm()._wl_sym_free(this._handle)
      this._handle = null
    }
    this._freeFamilyEngine()
    this._fitted = false
  }

  _freeFamilyEngine() {
    if (this._familyEngine && this._familyEngine.dispose) this._familyEngine.dispose()
    this._familyEngine = null
  }
}

class SymbolicRegressor extends BaseSymModel {
  constructor(params = {}) {
    super(params, 'regression')
  }
  static get typeId() {
    return TYPE_ID_REGRESSOR
  }
  static async create(params = {}) {
    return BaseSymModel._create(SymbolicRegressor, params)
  }
  static defaultSearchSpace() {
    return familySearchSpace({
      population: { type: 'int_uniform', low: 128, high: 1024 },
      generations: { type: 'int_uniform', low: 80, high: 400 },
      maxNodes: { type: 'int_uniform', low: 7, high: 63 },
      operatorSet: { type: 'categorical', values: ['full', 'smooth', 'basic'] },
      complexityPenalty: { type: 'log_uniform', low: 1e-5, high: 1e-2 },
      mutationRate: { type: 'uniform', low: 0.15, high: 0.55 },
      crossoverRate: { type: 'uniform', low: 0.35, high: 0.75 },
      islands: { type: 'categorical', values: [1, 2, 4] },
      broodSize: { type: 'categorical', values: [1, 2] },
      complexityHofSize: { type: 'categorical', values: [0, 64, 128] },
      finalSelector: { type: 'categorical', values: ['objective', 'loss'] },
      loss: { type: 'categorical', values: ['mse', 'mae', 'huber'] }
    })
  }
}

class SymbolicClassifier extends BaseSymModel {
  constructor(params = {}) {
    super(params, 'classification')
  }
  static get typeId() {
    return TYPE_ID_CLASSIFIER
  }
  static async create(params = {}) {
    return BaseSymModel._create(SymbolicClassifier, params)
  }

  predictProba(X) {
    this._ensureFitted()
    if (this._familyEngine) return this._familyEngine.predictProba(X)
    const { rows, cols, data } = normalizeX(X, this._nFeatures)
    if (cols !== this._nFeatures)
      throw new Error(`X has ${cols} columns, expected ${this._nFeatures}`)
    const wasm = getWasm()
    const xPtr = writeArrayToWasm(wasm, data, 8)
    const nClasses = this._nClasses || this._classes.length || 2
    const outPtr = wasm._malloc(rows * nClasses * 8)
    try {
      const ret = wasm._wl_sym_predict_proba(this._handle, xPtr, rows, cols, outPtr)
      if (ret !== 0) throw new Error(`Symbolic predictProba failed: ${getLastError()}`)
      return new Float64Array(wasm.HEAPF64.buffer, outPtr, rows * nClasses).slice()
    } finally {
      wasm._free(xPtr)
      wasm._free(outPtr)
    }
  }

  static defaultSearchSpace() {
    return familySearchSpace({
      population: { type: 'int_uniform', low: 128, high: 1024 },
      generations: { type: 'int_uniform', low: 80, high: 400 },
      maxNodes: { type: 'int_uniform', low: 7, high: 63 },
      operatorSet: { type: 'categorical', values: ['full', 'smooth', 'basic'] },
      complexityPenalty: { type: 'log_uniform', low: 1e-5, high: 1e-2 },
      mutationRate: { type: 'uniform', low: 0.15, high: 0.55 },
      crossoverRate: { type: 'uniform', low: 0.35, high: 0.75 },
      islands: { type: 'categorical', values: [1, 2, 4] },
      broodSize: { type: 'categorical', values: [1, 2] },
      complexityHofSize: { type: 'categorical', values: [0, 64, 128] },
      finalSelector: { type: 'categorical', values: ['objective', 'loss'] }
    })
  }
}

class FormulaTransformer extends BaseSymModel {
  constructor(params = {}) {
    super(params, 'transformer')
  }
  static get typeId() {
    return TYPE_ID_TRANSFORMER
  }
  static async create(params = {}) {
    return BaseSymModel._create(FormulaTransformer, params)
  }
  transform(X) {
    return this.predict(X)
  }
  fitTransform(X, y) {
    this.fit(X, y)
    return this.transform(X)
  }
  static defaultSearchSpace() {
    return {
      topK: { type: 'int_uniform', low: 2, high: 16 },
      population: { type: 'int_uniform', low: 128, high: 1024 },
      generations: { type: 'int_uniform', low: 80, high: 400 },
      maxNodes: { type: 'int_uniform', low: 7, high: 63 },
      operatorSet: { type: 'categorical', values: ['full', 'smooth', 'basic'] },
      complexityPenalty: { type: 'log_uniform', low: 1e-5, high: 1e-2 },
      islands: { type: 'categorical', values: [1, 2, 4] },
      broodSize: { type: 'categorical', values: [1, 2] },
      complexityHofSize: { type: 'categorical', values: [0, 64, 128] },
      finalSelector: { type: 'categorical', values: ['objective', 'loss'] }
    }
  }
}

function registerSymLoaders() {
  register(TYPE_ID_REGRESSOR.replace('@1', '@2'), (m, t, b) =>
    SymbolicRegressor._fromBundle(m, t, b)
  )
  register(TYPE_ID_CLASSIFIER.replace('@1', '@2'), (m, t, b) =>
    SymbolicClassifier._fromBundle(m, t, b)
  )
  register(TYPE_ID_REGRESSOR, (m, t, b) => SymbolicRegressor._fromBundle(m, t, b))
  register(TYPE_ID_CLASSIFIER, (m, t, b) => SymbolicClassifier._fromBundle(m, t, b))
  register(TYPE_ID_TRANSFORMER, (m, t, b) => FormulaTransformer._fromBundle(m, t, b))
}

registerSymLoaders()

module.exports = {
  SymbolicRegressor,
  SymbolicClassifier,
  FormulaTransformer,
  FormulaVerifier,
  registerSymLoaders,
  TYPE_ID_REGRESSOR,
  TYPE_ID_CLASSIFIER,
  TYPE_ID_TRANSFORMER
}

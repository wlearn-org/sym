'use strict'

function requirePolygrad() {
  if (typeof process !== 'undefined' && process.env && process.env.WLEARN_SYM_POLYGRAD_JS) {
    return require(process.env.WLEARN_SYM_POLYGRAD_JS + '/src/index.async.js')
  }
  return require('polygrad/async')
}

async function loadPolygrad(options) {
  if (options && typeof options.then === 'function') {
    return loadPolygrad(await options)
  }
  if (options && options.Tensor) return options
  // Construction is already asynchronous; let Polygrad resolve options and env.
  return requirePolygrad().createAsync(options)
}

function normalizeXForPolygrad(X, nFeatures) {
  if (X instanceof Float64Array || X instanceof Float32Array) {
    if (!nFeatures) throw new Error('nFeatures is required for flat typed-array input')
    if (X.length % nFeatures !== 0)
      throw new Error(`flat X length ${X.length} is not divisible by nFeatures ${nFeatures}`)
    return { rows: X.length / nFeatures, cols: nFeatures, data: new Float32Array(X) }
  }
  if (!Array.isArray(X))
    throw new Error('X must be an array of rows or a Float64Array/Float32Array')
  const rows = X.length
  const cols = rows > 0 && Array.isArray(X[0]) ? X[0].length : 1
  const data = new Float32Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    const row = Array.isArray(X[i]) ? X[i] : [X[i]]
    if (row.length !== cols)
      throw new Error(`X row ${i} has ${row.length} columns, expected ${cols}`)
    for (let j = 0; j < cols; j++) data[i * cols + j] = Number(row[j])
  }
  return { rows, cols, data }
}

function constTensor(Tensor, rows, value) {
  return Tensor.full([rows], Number(value), { dtype: 'float32', buffer: false })
}

function buildFormulaTensor(Tensor, XTensor, formula, rows) {
  return buildFormulaTensorInternal(Tensor, XTensor, formula, rows, null)
}

function buildFormulaTensorInternal(Tensor, XTensor, formula, rows, constParams) {
  const values = []
  for (let i = 0; i < formula.nodes.length; i++) {
    const node = formula.nodes[i]
    let out
    switch (node.op) {
      case 'const':
        out =
          constParams && constParams.has(i)
            ? constParams.get(i)
            : constTensor(Tensor, rows, node.value)
        break
      case 'var':
        out = XTensor.getitem([0, rows], node.feature)
        break
      case 'add':
        out = values[node.left].add(values[node.right])
        break
      case 'sub':
        out = values[node.left].sub(values[node.right])
        break
      case 'mul':
        out = values[node.left].mul(values[node.right])
        break
      case 'div':
        {
          const denRaw = values[node.right]
          const nearZero = denRaw.abs().lt(1e-12)
          const signedEps = denRaw.ge(0).where(1e-12, -1e-12)
          out = values[node.left].div(nearZero.where(signedEps, denRaw))
        }
        break
      case 'neg':
        out = values[node.left].neg()
        break
      case 'abs':
        out = values[node.left].abs()
        break
      case 'sqrt':
        out = values[node.left].abs().sqrt()
        break
      case 'log':
        out = values[node.left].abs().add(1e-12).log()
        break
      case 'exp':
        out = values[node.left].maximum(-40).minimum(40).exp()
        break
      case 'sin':
        out = values[node.left].sin()
        break
      case 'cos':
        out = values[node.left].cos()
        break
      case 'tanh':
        out = values[node.left].tanh()
        break
      case 'min':
        out = values[node.left].minimum(values[node.right])
        break
      case 'max':
        out = values[node.left].maximum(values[node.right])
        break
      default:
        throw new Error(`unsupported formula op for Polygrad: ${node.op}`)
    }
    // Match the C evaluator after every node, in float32 arithmetic.
    values.push(out.isfinite().where(out, 0).maximum(-1e12).minimum(1e12))
  }
  return values[values.length - 1]
}

async function evaluateFormulaPolygrad(runtime, formula, X, nFeatures) {
  const { rows, cols, data } = normalizeXForPolygrad(X, nFeatures)
  if (cols !== nFeatures) throw new Error(`X has ${cols} columns, expected ${nFeatures}`)
  const { Tensor } = runtime
  if (!Tensor) throw new Error('polygrad runtime is missing Tensor')
  const x = Tensor.empty([rows, cols], { dtype: 'float32' })
  let model
  try {
    model = await runtime.Model.fromCallableAsync(
      ({ sym_x }) => buildFormulaTensor(Tensor, sym_x, formula, rows),
      {
        inputs: { sym_x: x }
      }
    )
    const result = await model.forwardAsync({ sym_x: data })
    return new Float64Array(result.output)
  } finally {
    if (model) model.dispose()
    x.dispose()
  }
}

const REFINABLE_OPS = new Set([
  'const',
  'var',
  'add',
  'sub',
  'mul',
  'div',
  'neg',
  'abs',
  'sqrt',
  'log',
  'sin',
  'cos',
  'tanh',
  'exp',
  'min',
  'max'
])

function unsupportedRefineOp(formula) {
  for (const node of formula.nodes) {
    if (!REFINABLE_OPS.has(node.op)) return node.op
  }
  return null
}

function mseFloat64(y, pred) {
  let loss = 0
  for (let i = 0; i < y.length; i++) {
    const e = pred[i] - y[i]
    loss += e * e
  }
  return loss / Math.max(1, y.length)
}

async function refineFormulaPolygrad(runtime, formula, X, y, nFeatures, opts = {}) {
  const unsupported = unsupportedRefineOp(formula)
  if (unsupported) {
    return {
      status: 'unsupported',
      reason: `operator '${unsupported}' is not in the differentiable Polygrad refinement subset`,
      committedConstants: []
    }
  }

  const constNodes = []
  for (let i = 0; i < formula.nodes.length; i++) {
    if (formula.nodes[i].op === 'const') constNodes.push(i)
  }
  if (!constNodes.length) {
    return {
      status: 'unchanged',
      reason: 'formula has no constants to refine',
      committedConstants: []
    }
  }

  const { rows, cols, data } = normalizeXForPolygrad(X, nFeatures)
  if (cols !== nFeatures) throw new Error(`X has ${cols} columns, expected ${nFeatures}`)
  const yData = y instanceof Float32Array ? y : new Float32Array(y)
  if (yData.length !== rows)
    throw new Error(`y length (${yData.length}) does not match X rows (${rows})`)

  const { Tensor, Model } = runtime
  const epochs = opts.epochs ?? opts.steps ?? 80
  const lr = opts.lr ?? 0.001
  if (!Number.isSafeInteger(epochs) || epochs < 1 || !Number.isFinite(lr) || lr <= 0) {
    throw new Error('epochs must be positive integer and lr positive finite')
  }
  const params = new Map(),
    paramObject = {}
  let x, target, model
  try {
    x = Tensor.empty([rows, cols], { dtype: 'float32' })
    target = Tensor.empty([rows], { dtype: 'float32' })
    for (const i of constNodes) {
      const tensor = new Tensor([Number(formula.nodes[i].value)], { dtype: 'float32' })
      params.set(i, tensor)
      paramObject[`c${i}`] = tensor
    }
    model = await Model.fromCallableAsync(
      ({ sym_refine_x }) => buildFormulaTensorInternal(Tensor, sym_refine_x, formula, rows, params),
      {
        inputs: { sym_refine_x: x },
        targets: { sym_refine_y: target },
        params: paramObject,
        loss: (pred, { sym_refine_y }) => pred.sub(sym_refine_y).square().mean()
      }
    )
    const history = await model.fitAsync(
      { sym_refine_x: data, sym_refine_y: yData },
      {
        epochs,
        optimizer: opts.optimizer || 'adam',
        lr
      }
    )
    const refinedFormula = JSON.parse(JSON.stringify(formula)),
      committedConstants = []
    for (const i of constNodes) {
      // Read trained Model state, not the tensors originally captured by the graph.
      const value = Number((await model.readBufferAsync(`c${i}`))[0])
      if (!Number.isFinite(value))
        throw new Error('Polygrad refinement produced a non-finite constant')
      refinedFormula.nodes[i].value = value
      committedConstants.push({ nodeIndex: i, value })
    }
    const pred = await evaluateFormulaPolygrad(runtime, refinedFormula, X, nFeatures)
    return {
      status: 'ok',
      loss: mseFloat64(yData, pred),
      history,
      formula: refinedFormula,
      committedConstants
    }
  } finally {
    if (model) model.dispose()
    for (const tensor of [x, target, ...params.values()]) if (tensor) tensor.dispose()
  }
}

module.exports = { loadPolygrad, evaluateFormulaPolygrad, refineFormulaPolygrad }

'use strict'
const { ValidationError } = require('@wlearn/core')
const { loadPolygrad } = require('./polygrad.js')

// The structure is fixed. Only active nonlinear constants, linear coefficients
// and the intercept are trainable; validation rows are removed by the C owner.
function graph(X, c, cols, params, clipParameters = true) {
  const values = []
  let result = params.bias
  for (let t = 0; t < c.terms; t++) {
    const source = index =>
      index < cols ? X.getitem([0, X.shape[0]], index) : values[index - cols]
    const a = source(c.fa[t]),
      b = source(c.fb[t]),
      op = c.op[t]
    let value
    if (op === 0) value = a.add(b)
    else if (op === 1) value = a.sub(b)
    else if (op === 2) value = a.mul(b)
    else if (op === 3) value = a.div(b.abs().lt(1e-6).where(b.lt(0).where(-1e-6, 1e-6), b))
    else {
      let p0 = params[`p0_${t}`],
        p1 = params[`p1_${t}`]
      if (clipParameters) {
        p0 = p0.maximum(-16).minimum(16)
        p1 = p1.maximum(-16).minimum(16)
      }
      const z = a.mul(p0).add(p1)
      if (op === 4) value = z.sin()
      else if (op === 5) value = z.cos()
      else if (op === 6) value = z.tanh()
      else if (op === 7) value = z.abs().add(1e-6).log()
      else if (op === 8) value = z.abs().sqrt()
      else value = z.maximum(-6).minimum(6).exp()
    }
    values.push(value)
    result = result.add(params[`coef_${t}`].mul(value))
  }
  return result
}
async function refine(candidate, X, target, opts) {
  const epochs = opts.epochs ?? 80,
    lr = opts.lr ?? 0.001
  if (!Number.isInteger(epochs) || epochs < 1 || !Number.isFinite(lr) || lr <= 0)
    throw new ValidationError('epochs must be positive integer and lr positive finite')
  const options = await (opts.polygradRuntime || opts.polygrad)
  const pg = await loadPolygrad(options)
  const owned = !(options && options.Tensor)
  const params = {}
  let x, y, model
  try {
    for (let t = 0; t < candidate.terms; t++) {
      const keys = candidate.op[t] >= 4 ? ['p0', 'p1', 'coef'] : ['coef']
      for (const key of keys) {
        const tensor = pg.Tensor.empty([1], { dtype: 'float32' })
        params[`${key}_${t}`] = tensor
        await tensor.copyFromAsync(Float32Array.of(candidate[key][t]))
      }
    }
    params.bias = pg.Tensor.empty([1], { dtype: 'float32' })
    await params.bias.copyFromAsync(Float32Array.of(candidate.bias))
    x = pg.Tensor.empty([X.rows, X.cols], { dtype: 'float32' })
    y = pg.Tensor.empty([X.rows], { dtype: 'float32' })
    model = await pg.Model.fromCallableAsync(
      ({ family_x }) => graph(family_x, candidate, X.cols, params),
      {
        inputs: { family_x: x },
        targets: { family_y: y },
        params,
        loss: (pred, { family_y }) => {
          let penalty = params.coef_0.square().sum()
          for (let t = 1; t < candidate.terms; t++)
            penalty = penalty.add(params[`coef_${t}`].square().sum())
          return pred
            .sub(family_y)
            .square()
            .mean()
            .add(penalty.mul(opts.ridge ?? 1e-8))
        }
      }
    )
    const history = await model.fitAsync(
      { family_x: Float32Array.from(X.data), family_y: Float32Array.from(target) },
      {
        epochs,
        lr,
        optimizer: opts.optimizer || 'adam'
      }
    )
    const updates = new Float64Array(3 * candidate.terms + 1)
    for (let t = 0; t < candidate.terms; t++) {
      updates[3 * t] =
        candidate.op[t] >= 4 ? Number((await model.readBufferAsync(`p0_${t}`))[0]) : 1
      updates[3 * t + 1] =
        candidate.op[t] >= 4 ? Number((await model.readBufferAsync(`p1_${t}`))[0]) : 0
      updates[3 * t + 2] = Number((await model.readBufferAsync(`coef_${t}`))[0])
    }
    updates[3 * candidate.terms] = Number((await model.readBufferAsync('bias'))[0])
    return { updates, history }
  } finally {
    if (model) model.dispose()
    for (const t of [x, y, ...Object.values(params)]) if (t) t.dispose()
    if (owned) pg.dispose()
  }
}
async function predict(candidate, X, options) {
  options = await options
  if (!X.rows) return new Float64Array()
  const { checkPhase } = require('./family-scorer.js')
  const pg = await loadPolygrad(options)
  const owned = !(options && options.Tensor)
  const params = {}
  let x, model
  try {
    const dtype = pg.caps.f64 ? 'float64' : 'float32'
    const ArrayType = dtype === 'float64' ? Float64Array : Float32Array
    const data = ArrayType.from(X.data, Math.fround)
    const bounds = new Float64Array(X.cols)
    for (let i = 0; i < data.length; i++)
      bounds[i % X.cols] = Math.max(bounds[i % X.cols], Math.abs(data[i]))
    const descriptors = Float64Array.from(
      candidate.op.flatMap((op, t) => [
        candidate.fa[t],
        candidate.fb[t],
        op,
        candidate.p0[t],
        candidate.p1[t]
      ])
    )
    checkPhase(bounds, descriptors, 1, candidate.terms, dtype)
    for (let t = 0; t < candidate.terms; t++) {
      for (const key of candidate.op[t] >= 4 ? ['p0', 'p1', 'coef'] : ['coef']) {
        const tensor = pg.Tensor.empty([1], { dtype })
        params[`${key}_${t}`] = tensor
        await tensor.copyFromAsync(ArrayType.of(candidate[key][t]))
      }
    }
    params.bias = pg.Tensor.empty([1], { dtype })
    await params.bias.copyFromAsync(ArrayType.of(candidate.bias))
    x = pg.Tensor.empty([X.rows, X.cols], { dtype })
    model = await pg.Model.fromCallableAsync(
      ({ family_x }) => graph(family_x, candidate, X.cols, params, false),
      { inputs: { family_x: x }, params }
    )
    const result = Float64Array.from((await model.forwardAsync({ family_x: data })).output)
    if (!result.every(Number.isFinite))
      throw new ValidationError('nonfinite Polygrad family prediction')
    return result
  } finally {
    if (model) model.dispose()
    for (const t of [x, ...Object.values(params)]) if (t) t.dispose()
    if (owned) pg.dispose()
  }
}
module.exports = { refine, predict }

'use strict'

const { ValidationError } = require('@wlearn/core')
const { loadPolygrad } = require('./polygrad.js')

function evaluateTerm(op, a, v, z) {
  const abs = x => x.lt(0).where(x.neg(), x)
  const clamp = (x, lo, hi) => x.lt(lo).where(lo, x.lt(hi).where(x, hi))
  const den = abs(v).lt(1e-6).where(v.lt(0).where(-1e-6, 1e-6), v)
  const values = [
    a.add(v),
    a.sub(v),
    a.mul(v),
    a.div(den),
    z.sin(),
    z.add(Math.PI / 2).sin(),
    clamp(z, -20, 20)
      .mul(-2 / Math.LN2)
      .exp2()
      .add(1)
      .reciprocal()
      .mul(2)
      .sub(1),
    abs(z).add(1e-6).log2().mul(Math.LN2),
    abs(z).sqrt(),
    clamp(z, -6, 6).div(Math.LN2).exp2()
  ]
  let value = values[0]
  for (let j = 1; j < 10; j++) value = op.eq(j).where(values[j], value)
  return value
}

// One retained graph per stage/shape; the runtime owns device data. C remains
// the owner of the candidate stream, target masks, ridge, and acceptance.
class FamilyScorer {
  static async create(X, target, training, ym, descriptors, batch, terms, solve, params) {
    const s = new FamilyScorer()
    s.stages = []
    s.buffers = []
    s.outputs = new Set()
    const options = await (params.polygradRuntime || params.polygrad)
    s.owned = !(options && options.Tensor)
    s.pg = await loadPolygrad(options)
    try {
      s.rows = X.rows
      s.cols = X.cols
      s.batch = batch
      s.terms = terms
      s.hierarchical = !!params.hierarchical
      s.width = 2 ** Math.ceil(Math.log2(terms + 1))
      s.ym = ym
      s.solve = solve
      s.calls = 0
      s.dtype = params.scorerDtype || (s.pg.caps.f64 ? 'float64' : 'float32')
      if (!['float32', 'float64'].includes(s.dtype))
        throw new ValidationError('scorerDtype must be float32 or float64')
      if (s.dtype === 'float64' && !s.pg.caps.f64)
        throw new ValidationError('selected Polygrad device does not support float64')
      s.ArrayType = s.dtype === 'float64' ? Float64Array : Float32Array
      s.nt = training.reduce((a, b) => a + b, 0)
      s.nv = X.rows - s.nt
      s.anchor = training.findIndex(v => v === 1)
      s.bounds = new Float64Array(X.cols)
      const x = Float32Array.from(X.data)
      for (let i = 0; i < x.length; i++)
        s.bounds[i % X.cols] = Math.max(s.bounds[i % X.cols], Math.abs(x[i]))
      s.checkPhase(descriptors)
      s.X = await s.buffer([X.rows, X.cols], s.ArrayType.from(x))
      s.y = await s.buffer([X.rows], s.ArrayType.from(target))
      s.mask = await s.buffer([X.rows], s.ArrayType.from(training))
      const [indices, constants] = s.pack(descriptors)
      s.indices = await s.buffer([batch, terms, 3], indices, 'int32')
      s.constants = await s.buffer([batch, terms, 2], constants)
      s.coef = await s.buffer([batch, terms + 1], new s.ArrayType(batch * (terms + 1)))
      s.feature = await s.compile(
        (a, b, c) => (s.hierarchical ? s.hierarchicalFeatures(a, b, c) : s.features(a, b, c)),
        [s.X, s.indices, s.constants]
      )
      const phi = await s.feature.run([s.X, s.indices, s.constants])
      s.normalize = await s.compile((a, b, c) => s.normalized(a, b, c), [phi, s.y, s.mask])
      const [design] = await s.normalize.run([phi, s.y, s.mask])
      s.factor = await s.compile(a => a.qr('r'), [design])
      s.loss = await s.compile((a, b, c, d) => s.losses(a, b, c, d), [phi, s.y, s.mask, s.coef])
      return s
    } catch (err) {
      s.dispose()
      throw err
    }
  }
  async buffer(shape, data, dtype = this.dtype) {
    const t = this.pg.Tensor.empty(shape, { dtype })
    this.buffers.push(t)
    await t.copyFromAsync(data)
    return t
  }
  async compile(fn, inputs) {
    const stage = await this.pg.compileAsync((...args) => {
      const result = fn(...args)
      for (const tensor of Array.isArray(result) ? result : [result]) this.outputs.add(tensor)
      return result
    }, inputs)
    this.stages.push(stage)
    return stage
  }
  pack(descriptors) {
    const indices = new Int32Array(this.batch * this.terms * 3)
    const constants = new this.ArrayType(this.batch * this.terms * 2)
    for (let i = 0; i < this.batch * this.terms; i++) {
      indices.set(descriptors.subarray(5 * i, 5 * i + 3), 3 * i)
      constants[2 * i] = Math.fround(descriptors[5 * i + 3])
      constants[2 * i + 1] = Math.fround(descriptors[5 * i + 4])
    }
    return [indices, constants]
  }
  checkPhase(d) {
    checkPhase(this.bounds, d, this.batch, this.terms, this.dtype)
  }
  features(X, indices, constants) {
    const { pg, rows: n, cols, batch: b, terms: t } = this
    const kernel = (out, x, di, df) => {
      out = out.flatten()
      x = x.flatten()
      di = di.flatten()
      df = df.flatten()
      const i = pg.uop.range(b * n * t, 0)
      const c = i.floordiv(n * t),
        row = i.floordiv(t).mod(n),
        term = i.mod(t)
      const at = c.mul(t).add(term),
        ix = at.mul(3),
        fp = at.mul(2)
      const a = x.index(row.mul(cols).add(di.index(ix)))
      const v = x.index(row.mul(cols).add(di.index(ix.add(1))))
      const op = di.index(ix.add(2)),
        z = a.mul(df.index(fp)).add(df.index(fp.add(1)))
      const value = evaluateTerm(op, a, v, z)
      return out.index(i).store(value).end(i).sink(new pg.uop.KernelInfo('sym_family_terms'))
    }
    return pg.Tensor.empty([b, n, t], { dtype: this.dtype }).customKernel(
      X,
      indices,
      constants,
      kernel
    )[0]
  }
  hierarchicalFeatures(X, indices, constants) {
    const { pg, batch: b, rows: n, terms: t, cols } = this
    const kernel = (out, x, di, df) => {
      out = out.flatten()
      x = x.flatten()
      di = di.flatten()
      df = df.flatten()
      const i = pg.uop.range(b * n, 0),
        c = i.floordiv(n),
        row = i.mod(n)
      const values = [],
        stores = []
      for (let j = 0; j < t; j++) {
        const at = c.mul(t).add(j),
          ix = at.mul(3),
          fp = at.mul(2)
        const source = index => {
          let value = x.index(row.mul(cols).add(index.lt(cols).where(index, 0)))
          for (let earlier = 0; earlier < values.length; earlier++)
            value = index.eq(cols + earlier).where(values[earlier], value)
          return value
        }
        const a = source(di.index(ix)),
          v = source(di.index(ix.add(1)))
        const value = evaluateTerm(
          di.index(ix.add(2)),
          a,
          v,
          a.mul(df.index(fp)).add(df.index(fp.add(1)))
        )
        values.push(value)
        stores.push(out.index(i.mul(t).add(j)).store(value))
      }
      return stores[0]
        .group(...stores.slice(1))
        .end(i)
        .sink(new pg.uop.KernelInfo('sym_family_hierarchy'))
    }
    return pg.Tensor.empty([b, n, t], { dtype: this.dtype }).customKernel(
      X,
      indices,
      constants,
      kernel
    )[0]
  }
  normalized(phi, y, mask) {
    const { batch: b, rows: n, terms: t, width: k, anchor: a, nt, ym } = this
    const anchor = phi.getitem([0, b], [a, a + 1], [0, t])
    const delta = phi.sub(anchor),
      train = mask.reshape(1, n, 1)
    const mean = delta.mul(train).sum(1, true).div(nt)
    const centered = delta.sub(mean)
    let scale = centered.square().mul(train).sum(1, true).div(nt).sqrt()
    scale = scale.eq(0).where(1, scale)
    const target = y.sub(ym).reshape(1, n, 1).expand(b, n, 1)
    let design = centered.div(scale).cat(target, { dim: 2 }).mul(train)
    design = design.pad([
      [0, 0],
      [0, Math.max(0, k - n)],
      [0, k - t - 1]
    ])
    const stats = anchor.cat(mean, scale, { dim: 2 }).reshape(b, 3 * t)
    return [design.contiguous(), stats.contiguous()]
  }
  losses(phi, y, mask, coef, residuals = false) {
    const { pg, batch: b, rows: n, terms: t, anchor, nt, nv } = this
    const kernel = (out, features, target, train, weights) => {
      out = out.flatten()
      features = features.flatten()
      target = target.flatten()
      train = train.flatten()
      weights = weights.flatten()
      const c = pg.uop.range(b, 0),
        r = pg.uop.range(n, 1, residuals ? pg.uop.AxisType.LOOP : pg.uop.AxisType.REDUCE)
      let pred = weights.index(c.mul(t + 1).add(t))
      for (let j = 0; j < t; j++) {
        const value = features.index(c.mul(n).add(r).mul(t).add(j))
        const shift = features.index(c.mul(n).add(anchor).mul(t).add(j))
        pred = pred.add(value.sub(shift).mul(weights.index(c.mul(t + 1).add(j))))
      }
      const delta = pred.sub(target.index(r)),
        squared = delta.mul(delta)
      const m = train.index(r)
      if (residuals)
        return out.index(c.mul(n).add(r)).store(delta.mul(m)).end(r, c)
          .sink(new pg.uop.KernelInfo('sym_family_residuals'))
      const first = out.index(c.mul(2)).store(squared.mul(m).sum(r).div(nt))
      const second = out
        .index(c.mul(2).add(1))
        .store(squared.mul(m.neg().add(1)).sum(r).div(Math.max(1, nv)))
      return first.group(second).end(c).sink(new pg.uop.KernelInfo('sym_family_losses'))
    }
    return pg.Tensor.empty([b, residuals ? n : 2], { dtype: this.dtype }).customKernel(
      phi,
      y,
      mask,
      coef,
      kernel
    )[0]
  }
  async score(descriptors, needResiduals = false) {
    if (!this.pg) throw new ValidationError('family scorer is disposed')
    this.checkPhase(descriptors)
    const [indices, constants] = this.pack(descriptors)
    await this.indices.copyFromAsync(indices)
    await this.constants.copyFromAsync(constants)
    const phi = await this.feature.run([this.X, this.indices, this.constants])
    const [design, stats] = await this.normalize.run([phi, this.y, this.mask])
    const factors = await this.factor.run([design])
    const [hs, hf] = await this.pg.Tensor.toTypedArraysAsync(stats, factors)
    if (!hs.every(Number.isFinite) || !hf.every(Number.isFinite))
      throw new ValidationError('nonfinite device statistics or QR factors')
    const coefficients = this.solve(hf, hs, this.width)
    const shifted = this.ArrayType.from(coefficients)
    for (let c = 0; c < this.batch; c++) {
      let bias = coefficients[c * (this.terms + 1) + this.terms]
      for (let t = 0; t < this.terms; t++)
        bias += coefficients[c * (this.terms + 1) + t] * hs[c * 3 * this.terms + t]
      shifted[c * (this.terms + 1) + this.terms] = bias
    }
    await this.coef.copyFromAsync(shifted)
    const losses = await (
      await this.loss.run([phi, this.y, this.mask, this.coef])
    ).toTypedArrayAsync()
    if (!losses.every(Number.isFinite)) throw new ValidationError('nonfinite device losses')
    if (!this.nv) for (let c = 0; c < this.batch; c++) losses[2 * c + 1] = losses[2 * c]
    let residuals
    if (needResiduals) {
      if (!this.residual)
        this.residual = await this.compile((a, b, c, d) => this.losses(a, b, c, d, true),
          [phi, this.y, this.mask, this.coef])
      residuals = await (await this.residual.run([phi, this.y, this.mask, this.coef])).toTypedArrayAsync()
    }
    if (++this.calls % 32 === 0) this.pg.collect()
    return { coefficients, losses, residuals }
  }
  dispose() {
    for (const stage of this.stages.reverse()) stage.dispose()
    // Compiled stages replay the same output handles; release them only after
    // the stages, including the separate uncaptured warmup outputs.
    for (const output of this.outputs) output.dispose()
    this.outputs.clear()
    for (const buffer of this.buffers) buffer.dispose()
    this.stages = []
    this.buffers = []
    if (this.owned && this.pg) this.pg.dispose()
    this.pg = null
  }
}
function checkPhase(featureBounds, d, batch, terms, dtype) {
  for (let c = 0; c < batch; c++) {
    const bounds = Array.from(featureBounds)
    for (let t = 0; t < terms; t++) {
      const i = (c * terms + t) * 5
      const a = bounds[d[i]],
        b = bounds[d[i + 1]],
        op = d[i + 2]
      const z = a * Math.abs(d[i + 3]) + Math.abs(d[i + 4])
      if (op === 5) {
        const ulp = 2 ** (Math.floor(Math.log2(z)) - (dtype === 'float64' ? 52 : 23))
        if (!Number.isFinite(z) || ulp > 2e-6)
          throw new ValidationError('cosine phase exceeds the scorer precision budget; use C')
      }
      bounds.push(
        [
          a + b,
          a + b,
          a * b,
          a / 1e-6,
          1,
          1,
          1,
          Math.max(Math.abs(Math.log(1e-6)), Math.abs(Math.log(z + 1e-6))),
          Math.sqrt(z),
          Math.exp(6)
        ][op]
      )
    }
  }
}

module.exports = { FamilyScorer, checkPhase }

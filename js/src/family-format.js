'use strict'

const OPS = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'tanh', 'logabs', 'sqrtabs', 'expclamp']
const OP_PRESETS = {
  basic: ['add', 'sub', 'mul', 'div'],
  smooth: ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'tanh'],
  full: OPS
}
const OP_INDEX = new Map(OPS.map((name, index) => [name, index]))

function resolveOperatorIds(params = {}) {
  const requested = params.operators || params.operatorSet || params.operator_set || 'full'
  const names = Array.isArray(requested)
    ? requested
    : OP_PRESETS[String(requested).toLowerCase()] || null
  if (!names) throw new Error(`unsupported pg-family operatorSet: ${requested}`)
  const ids = []
  for (const value of names) {
    const id = typeof value === 'number' ? value : OP_INDEX.get(String(value).toLowerCase())
    if (!Number.isInteger(id) || id < 0 || id >= OPS.length) {
      throw new Error(`unsupported pg-family operator: ${value}`)
    }
    if (!ids.includes(id)) ids.push(id)
  }
  if (!ids.length) throw new Error('pg-family operators must not be empty')
  return ids
}

function formulaText(c, featureNames) {
  const parts = [],
    expressions = []
  const hierarchical = [...c.fa, ...c.fb].some(i => i >= featureNames.length)
  for (let t = 0; t < c.terms; t++) {
    const a =
      c.fa[t] < featureNames.length ? featureNames[c.fa[t]] : `t${c.fa[t] - featureNames.length}`
    const b =
      c.fb[t] < featureNames.length ? featureNames[c.fb[t]] : `t${c.fb[t] - featureNames.length}`
    const z = `${Number(c.p0[t]).toPrecision(5)}*${a}+${Number(c.p1[t]).toPrecision(5)}`
    let expr
    if (c.op[t] === 0) expr = `(${a}+${b})`
    else if (c.op[t] === 1) expr = `(${a}-${b})`
    else if (c.op[t] === 2) expr = `(${a}*${b})`
    else if (c.op[t] === 3) expr = `protected_div(${a},${b})`
    else if (c.op[t] === 4) expr = `sin(${z})`
    else if (c.op[t] === 5) expr = `cos(${z})`
    else if (c.op[t] === 6) expr = `tanh(${z})`
    else if (c.op[t] === 7) expr = `log(abs(${z})+1e-6)`
    else if (c.op[t] === 8) expr = `sqrt(abs(${z}))`
    else expr = `exp(clamp(${z},-6,6))`
    expressions.push(expr)
    parts.push(`${Number(c.coef[t]).toPrecision(5)}*${hierarchical ? `t${t}` : expr}`)
  }
  const result = `${Number(c.bias).toPrecision(5)} + ${parts.join(' + ')}`
  return hierarchical
    ? `${expressions.map((v, i) => `t${i} = ${v}`).join('; ')}; ${result}`
    : result
}

module.exports = { OPS, resolveOperatorIds, formulaText }

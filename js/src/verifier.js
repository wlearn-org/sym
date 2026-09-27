'use strict'

function coeffZero(n) {
  return new Array(n).fill(0)
}

function affine(formula, nFeatures) {
  const vals = []
  for (const node of formula.nodes) {
    if (node.op === 'const') {
      vals.push({ bias: Number(node.value), coeff: coeffZero(nFeatures) })
    } else if (node.op === 'var') {
      const coeff = coeffZero(nFeatures)
      coeff[node.feature] = 1
      vals.push({ bias: 0, coeff })
    } else if (node.op === 'add' || node.op === 'sub') {
      const a = vals[node.left], b = vals[node.right]
      if (!a || !b) return null
      vals.push({
        bias: node.op === 'add' ? a.bias + b.bias : a.bias - b.bias,
        coeff: a.coeff.map((v, i) => node.op === 'add' ? v + b.coeff[i] : v - b.coeff[i])
      })
    } else if (node.op === 'neg') {
      const a = vals[node.left]
      if (!a) return null
      vals.push({ bias: -a.bias, coeff: a.coeff.map(v => -v) })
    } else if (node.op === 'mul') {
      const a = vals[node.left], b = vals[node.right]
      if (!a || !b) return null
      const aConst = a.coeff.every(v => Math.abs(v) < 1e-15)
      const bConst = b.coeff.every(v => Math.abs(v) < 1e-15)
      if (aConst) vals.push({ bias: a.bias * b.bias, coeff: b.coeff.map(v => v * a.bias) })
      else if (bConst) vals.push({ bias: a.bias * b.bias, coeff: a.coeff.map(v => v * b.bias) })
      else return null
    } else if (node.op === 'div') {
      const a = vals[node.left], b = vals[node.right]
      if (!a || !b || !b.coeff.every(v => Math.abs(v) < 1e-15) || Math.abs(b.bias) < 1e-12) return null
      vals.push({ bias: a.bias / b.bias, coeff: a.coeff.map(v => v / b.bias) })
    } else {
      return null
    }
  }
  return vals[vals.length - 1] || null
}

function evalFormula(formula, row) {
  const vals = []
  for (const node of formula.nodes) {
    const a = node.left >= 0 ? vals[node.left] : 0
    const b = node.right >= 0 ? vals[node.right] : 0
    let v = 0
    switch (node.op) {
      case 'const': v = Number(node.value); break
      case 'var': v = row[node.feature] || 0; break
      case 'add': v = a + b; break
      case 'sub': v = a - b; break
      case 'mul': v = a * b; break
      case 'div': v = a / (Math.abs(b) < 1e-12 ? (b < 0 ? -1e-12 : 1e-12) : b); break
      case 'neg': v = -a; break
      case 'abs': v = Math.abs(a); break
      case 'sqrt': v = Math.sqrt(Math.abs(a)); break
      case 'log': v = Math.log(Math.abs(a) + 1e-12); break
      case 'exp': v = Math.exp(Math.max(-40, Math.min(40, a))); break
      case 'sin': v = Math.sin(a); break
      case 'cos': v = Math.cos(a); break
      case 'tanh': v = Math.tanh(a); break
      case 'min': v = Math.min(a, b); break
      case 'max': v = Math.max(a, b); break
      default: v = 0
    }
    vals.push(Number.isFinite(v) ? Math.max(-1e12, Math.min(1e12, v)) : 0)
  }
  return vals[vals.length - 1]
}

function normalizeFormula(formula) {
  if (typeof formula === 'string') return JSON.parse(formula)
  return formula
}

function numericMonotonic(formula, check, nFeatures) {
  const bounds = check.bounds || new Array(nFeatures).fill(0).map(() => [-2, 2])
  const feature = check.feature
  const direction = check.direction === 'decreasing' || check.direction === -1 ? -1 : 1
  const steps = check.steps || 9
  for (let t = 0; t < steps; t++) {
    const rowA = []
    const rowB = []
    for (let j = 0; j < nFeatures; j++) {
      const [lo, hi] = bounds[j] || [-2, 2]
      const v = lo + (hi - lo) * (((t * 1103515245 + j * 12345) & 0x7fffffff) / 0x7fffffff)
      rowA[j] = v
      rowB[j] = v
    }
    rowA[feature] = bounds[feature][0]
    rowB[feature] = bounds[feature][1]
    const a = evalFormula(formula, rowA)
    const b = evalFormula(formula, rowB)
    if (direction * (b - a) < -1e-8) {
      return { status: 'counterexample', check: 'monotonicity', counterexample: { low: rowA, high: rowB, lowValue: a, highValue: b } }
    }
  }
  return { status: 'unknown', check: 'monotonicity', reason: 'non-affine formula passed bounded numeric probes but was not exactly proved' }
}

class FormulaVerifier {
  static verify(formulaInput, checks = {}) {
    const formula = normalizeFormula(formulaInput)
    const nFeatures = checks.nFeatures || inferFeatureCount(formula)
    const requested = Array.isArray(checks) ? checks : (checks.checks || ['domain'])
    const results = []
    for (const checkName of requested) {
      const check = typeof checkName === 'string' ? { kind: checkName } : checkName
      if (check.kind === 'domain') {
        results.push({
          status: 'proved',
          check: 'domain',
          reason: 'all non-total operators in @wlearn/sym use protected semantics in the C core'
        })
      } else if (check.kind === 'monotonicity') {
        const a = affine(formula, nFeatures)
        const direction = check.direction === 'decreasing' || check.direction === -1 ? -1 : 1
        if (a) {
          const coeff = a.coeff[check.feature]
          results.push(direction * coeff >= -1e-12
            ? { status: 'proved', check: 'monotonicity', coefficient: coeff }
            : { status: 'counterexample', check: 'monotonicity', coefficient: coeff })
        } else {
          results.push(numericMonotonic(formula, check, nFeatures))
        }
      } else if (check.kind === 'range') {
        results.push({ status: 'unknown', check: 'range', reason: 'range proof requires interval/Z3 support for this formula subset' })
      } else if (check.kind === 'equivalence') {
        const other = normalizeFormula(check.other)
        const a = affine(formula, nFeatures)
        const b = affine(other, nFeatures)
        if (a && b) {
          const sameBias = Math.abs(a.bias - b.bias) < 1e-12
          const sameCoeff = a.coeff.every((v, i) => Math.abs(v - b.coeff[i]) < 1e-12)
          results.push(sameBias && sameCoeff
            ? { status: 'proved', check: 'equivalence' }
            : { status: 'counterexample', check: 'equivalence' })
        } else {
          results.push({ status: 'unknown', check: 'equivalence', reason: 'non-affine equivalence is not labelled proved without SMT support' })
        }
      } else {
        results.push({ status: 'unsupported', check: check.kind || String(checkName) })
      }
    }
    return {
      status: results.every(r => r.status === 'proved') ? 'proved' : results.some(r => r.status === 'counterexample') ? 'counterexample' : 'unknown',
      results
    }
  }
}

function inferFeatureCount(formula) {
  let n = 0
  for (const node of formula.nodes || []) {
    if (node.op === 'var') n = Math.max(n, Number(node.feature) + 1)
  }
  return n
}

module.exports = { FormulaVerifier, evalFormula, affine }

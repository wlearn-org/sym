'use strict'
const { ValidationError } = require('@wlearn/core')

function resolveStrategy(params, task) {
  let legacy = null
  if (params.engine != null) {
    const engine = String(params.engine).toLowerCase()
    if (engine === 'pg-family') legacy = { strategy: 'family', backend: 'polygrad', legacy: true }
    else if (['c', 'wasm', 'c-wasm', 'auto'].includes(engine))
      legacy = { strategy: 'tree', backend: 'c', legacy: false }
    else throw new ValidationError(`unsupported sym engine: ${params.engine}`)
  }
  const strategy = params.strategy ?? legacy?.strategy ?? 'tree'
  const backend = params.backend ?? legacy?.backend ?? 'c'
  if (!['tree', 'family'].includes(strategy) || !['c', 'polygrad'].includes(backend))
    throw new ValidationError('strategy must be tree|family and backend must be c|polygrad')
  if (legacy && (strategy !== legacy.strategy || backend !== legacy.backend))
    throw new ValidationError('engine conflicts with strategy/backend')
  if (task === 'transformer' && strategy !== 'tree')
    throw new ValidationError('family strategy does not support FormulaTransformer')
  if (backend === 'polygrad' && !legacy?.legacy)
    throw new ValidationError(
      'shared-search Polygrad backend is not implemented yet; engine="pg-family" remains a separate legacy implementation'
    )
  return { strategy, backend, legacy: Boolean(legacy?.legacy) }
}
function familySearchSpace(tree) {
  const space = {
    strategy: { type: 'categorical', values: ['tree', 'family'] },
    backend: { type: 'categorical', values: ['c'] },
    ...tree
  }
  for (const key of ['maxNodes', 'broodSize', 'complexityHofSize', 'finalSelector', 'loss']) {
    if (space[key]) space[key] = { ...space[key], condition: { strategy: 'tree' } }
  }
  space.terms = { type: 'int_uniform', low: 2, high: 12, condition: { strategy: 'family' } }
  space.ridge = { type: 'log_uniform', low: 1e-8, high: 1e-2, condition: { strategy: 'family' } }
  space.immigrantRate = {
    type: 'categorical',
    values: [0, 0.1, 0.2, 0.3],
    condition: { strategy: 'family' }
  }
  return space
}
module.exports = { resolveStrategy, familySearchSpace }

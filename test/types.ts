import { SymbolicRegressor, FormulaTransformer, SymParams } from '../js/src'
const params: SymParams = { strategy: 'family', backend: 'c', terms: 4, ridge: 0.001 }
SymbolicRegressor.create(params)
SymbolicRegressor.create({ strategy: 'tree', backend: 'c' })
SymbolicRegressor.create({ engine: 'pg-family', strategy: 'family', backend: 'polygrad' })
FormulaTransformer.create({ strategy: 'tree', backend: 'c', topK: 3 })
SymbolicRegressor.create({ strategy: 'family', backend: 'polygrad' })
// @ts-expect-error Legacy tree alias conflicts with family strategy.
SymbolicRegressor.create({ engine: 'c', strategy: 'family' })
// @ts-expect-error Family formula features are not implemented.
FormulaTransformer.create({ strategy: 'family', backend: 'c' })

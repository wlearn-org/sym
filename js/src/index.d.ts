import type { DenseMatrix, MaybePromise, SearchSpace } from '@wlearn/types'

export type SymStrategy = 'tree' | 'family'
export type SymBackend = 'c' | 'polygrad'
export type SymInput = DenseMatrix | number[][] | Float32Array | Float64Array
export type SymTarget = number[] | Int32Array | Float32Array | Float64Array
export type SymSelection =
  | { strategy?: 'tree'; backend?: 'c'; engine?: 'c' | 'wasm' | 'c-wasm' | 'auto' }
  | { strategy: 'family'; backend?: 'c'; engine?: never }
  /** Temporary legacy search; this does not select a shared C-search scorer. */
  | { strategy?: 'family'; backend?: 'polygrad'; engine: 'pg-family' }

export interface SymControls {
  population?: number
  generations?: number
  seed?: number
  operatorSet?: 'basic' | 'smooth' | 'full' | string[]
  operators?: (string | number)[]
  eliteCount?: number
  frontierSize?: number
  validationFraction?: number
  complexityPenalty?: number
  mutationRate?: number
  crossoverRate?: number
  earlyStopRounds?: number
  islands?: number
  tol?: number
  classes?: number[]
  /** Family controls. Runtime validation rejects these where unsupported. */
  terms?: number
  ridge?: number
  immigrantRate?: number
  polishPasses?: number
  /** Tree controls. */
  maxNodes?: number
  maxDepth?: number
  topK?: number
  loss?: 'mse' | 'mae' | 'huber' | 'logloss'
  tournamentSize?: number
  migrationInterval?: number
  migrationCount?: number
  broodSize?: number
  rowSampleSize?: number
  localRefineInterval?: number
  localRefineCount?: number
  complexityHofSize?: number
  finalSelector?: 'objective' | 'loss' | 'score'
  wasm?: Record<string, unknown>
  polygrad?: Record<string, unknown>
  /** Additional tree/legacy controls and Python-style parameter aliases. */
  [key: string]: unknown
}
export type SymParams = SymControls & SymSelection
export interface SymCapabilities {
  classifier: boolean
  regressor: boolean
  transformer: boolean
  predictProba: boolean
  strategies: SymStrategy[]
  backends: SymBackend[]
  activeStrategy: SymStrategy
  activeBackend: SymBackend
  legacySearch: boolean
  polygradExecution: boolean
  polygradRefinement: boolean
  [key: string]: unknown
}

declare class BaseSymModel {
  protected constructor(params?: SymParams)
  readonly isFitted: boolean
  readonly nFeatures: number
  readonly nOutputs: number
  readonly nClasses: number
  readonly classes: number[]
  readonly capabilities: SymCapabilities
  /** C paths return this synchronously; legacy pg-family returns a Promise. */
  fit(X: SymInput, y: SymTarget): MaybePromise<this>
  predict(X: SymInput): MaybePromise<Float64Array>
  decisionFunction(X: SymInput): MaybePromise<Float64Array>
  score(X: SymInput, y: SymTarget): MaybePromise<number>
  formula(opts: { format: 'text'; index?: number }): string
  formula(opts?: { format?: 'json'; index?: number }): Record<string, unknown>
  frontier(opts?: { format?: 'json' | 'text' }): Record<string, unknown>[]
  verify(checks?: Record<string, unknown>): unknown
  predictPolygrad(X: SymInput, opts?: Record<string, unknown>): Promise<Float64Array>
  refinePolygrad(X: SymInput, y: SymTarget, opts?: Record<string, unknown>): Promise<Record<string, unknown>>
  getParams(): SymParams
  setParams(params: Partial<SymParams>): this
  save(path?: string): Uint8Array
  dispose(): void
}
export class SymbolicRegressor extends BaseSymModel {
  static readonly typeId: 'wlearn.sym.regressor@1'
  static create(params?: SymParams): Promise<SymbolicRegressor>
  static load(bytesOrPath: Uint8Array | ArrayBuffer | string): Promise<SymbolicRegressor>
  static defaultSearchSpace(): SearchSpace
}
export class SymbolicClassifier extends BaseSymModel {
  static readonly typeId: 'wlearn.sym.classifier@1'
  static create(params?: SymParams): Promise<SymbolicClassifier>
  static load(bytesOrPath: Uint8Array | ArrayBuffer | string): Promise<SymbolicClassifier>
  static defaultSearchSpace(): SearchSpace
  predictProba(X: SymInput): MaybePromise<Float64Array>
}
export class FormulaTransformer extends BaseSymModel {
  static readonly typeId: 'wlearn.sym.transformer@1'
  static create(params?: SymControls & Extract<SymSelection, { strategy?: 'tree' }>): Promise<FormulaTransformer>
  static load(bytesOrPath: Uint8Array | ArrayBuffer | string): Promise<FormulaTransformer>
  static defaultSearchSpace(): SearchSpace
  transform(X: SymInput): Float64Array
  fitTransform(X: SymInput, y: SymTarget): Float64Array
}
export class FormulaVerifier {
  static verify(formula: Record<string, unknown>, checks?: Record<string, unknown>): unknown
}
export function loadSym(options?: Record<string, unknown>): Promise<unknown>
export function registerSymLoaders(): void

# @wlearn/sym

Symbolic regression, classification and formula features for Node and browsers.
C11 search runs through WebAssembly. Optional Polygrad scoring uses its public
frontend, sharing a caller-supplied runtime when desired.

Install Sym, then add Polygrad if you need accelerated scoring:

```sh
npm install @wlearn/sym
npm install polygrad@0.6.0  # optional
```

```js
const { SymbolicRegressor } = require('@wlearn/sym')
const X = [[0, 1], [1, 2], [2, 3], [3, 4]]
const y = [1, 3, 5, 7]
const model = await SymbolicRegressor.create({
  strategy: 'family', backend: 'c', terms: 3,
  population: 32, generations: 10, seed: 42
})
model.fit(X, y)
const predictions = model.predict([[4, 5]])
const text = model.formula({ format: 'text' })
const bytes = model.save()
const restored = await SymbolicRegressor.load(bytes)
```

## API

`SymbolicRegressor`, `SymbolicClassifier` and `FormulaTransformer` share
`create`, `fit`, `predict`, `score`, `getParams`, `setParams`, `save`, `load`,
`dispose`, `capabilities` and `defaultSearchSpace`. Classifiers add
`predictProba`, `decisionFunction` and `classes`. FormulaTransformer adds
`transform`/`fitTransform`. Inputs may be rectangular arrays, core dense matrices,
or flat typed arrays with a fitted feature count.

`formula({format: 'json'|'text', index?})` exposes fitted expressions;
`frontier()` exposes the objective-sorted archive. `verify()` reports unsupported
checks explicitly; Python offers optional Z3-backed tree verification.
`save(path)` is Node-only and returns the bytes written; browsers use `save()`.
Loaders register with `@wlearn/core` for nested Pipeline/ensemble artifacts.

## Strategy and execution

- Default `strategy: 'tree', backend: 'c'`: regression, classification, Transformer.
- `strategy: 'family'`, `backend: 'c'|'polygrad'`: regression and classification.
- Tree+Polygrad search and family Transformer are unsupported.

C fits are synchronous. Family+Polygrad fits return a Promise; await them,
including when composing Pipelines. Fitted prediction, score, probability,
save and dispose remain synchronous and require no Polygrad. Explicit
`predictPolygrad(X, options)` returns a Promise.

```js
const accelerated = await SymbolicRegressor.create({
  strategy: 'family', backend: 'polygrad',
  terms: 6, seed: 42, validationFraction: 0.2,
  hierarchical: false, polishPasses: 2, polishBatchSize: 32,
  polygrad: { core: 'native', device: 'cuda' }
})
await accelerated.fit(X, y)
const report = await accelerated.refinePolygrad(X, y, { epochs: 20, lr: 0.001 })
```

In browsers use `polygrad: { core: 'wasm', device: 'webgpu' }`. Sym selects the
public async frontend. To share a runtime, create it through `polygrad/async`
with `await createAsync(...)`, then pass `polygrad: runtime`. Borrowed runtimes
are never disposed by Sym. Internally created runtimes are isolated and owned.
Call model `dispose()` when replacing models in long-running applications.

Family controls include `terms` (1–32, default 6), `population` (128),
`generations` (20), `eliteCount` (8), `islands` (1), `immigrantRate` (0),
`ridge` (1e-8), `polishPasses` (0), `polishBatchSize` (0), `hierarchical` (false).
`scaleAware` (default false) searches nonlinear constants in train-only
standardized coordinates while saving formulas in original input units.
`polishMethod` defaults to `coordinate`; `lm` selects finite-difference damped
least squares with QR readout refits (`polishPasses > 0`, `polishBatchSize = 0`).
LM requires both training and validation-objective improvement for acceptance.
Float32 residual differences can change LM steps and the resulting search path.
It adds O(rows × terms) host residual storage/transfer with Polygrad; feature evaluation
and QR factorization still execute on the selected device. These are experiments,
not default changes. Target scaling and the absolute complexity penalty are unchanged.

Optional `localRefineInterval` and `localRefineCount` (both default 0) run the same
polish before breeding every interval. Enable both with `polishPasses > 0`;
count is at most `min(population, 32)`. Final polish remains enabled separately
by `polishPasses`. This experimental schedule refits readouts with QR and works
with either scorer; it does not introduce gradient optimization or change defaults.
Positive polish batch sizes use independent proposals around a frozen base;
zero retains sequential coordinate polish. Hierarchical terms can reference
inputs or earlier terms. `operatorSet` accepts basic/smooth/full or a list;
`operators` accepts a subset of the ten portable built-ins.

Polygrad evaluates features, QR and losses for every candidate. C owns the
search and small regularized triangular solves. `batchSize` (default `min(population, 512)`) is a
transport setting; `scorerDtype` selects float32/float64 where supported. Native
CPU/CUDA defaults to float64; WebGPU uses float32. Numerical differences may
change near-tied selections; large unsupported cosine phases raise errors.
Compilation can dominate small fits, so C remains the default.

Family readouts minimize MSE plus ridge, including classification margin fitting.
The seeded validation holdout is excluded from readout fitting and gradient
training. Selection and family refinement acceptance use holdout MSE plus
complexity; use the same input rows/order for refinement. `index` selects a
multiclass head. Tree refinement remains regression-only with training-MSE
acceptance. Probabilities are sigmoid/softmax scores, not calibrated uncertainty.

The old `engine: 'pg-family'` spelling now selects the shared C search with
Polygrad scoring; its former frontend search has been removed. Legacy tuning
options such as `scoreMode` and `stackSummaries` do not configure the new scorer.
Prefer explicit strategy/backend. Changing either invalidates the fitted model.

## Persistence and development

Tree bundles use `@1`; flat families use `@2`; hierarchical families use `@3`.
Legacy JSON family `@1` artifacts remain readable without Polygrad, preserving
old metrics on re-save. Refit writes the current family format. The static class
`typeId` refers to tree models; `save()` writes the actual typeId.

C source is canonical under the repository's `src/`; `csrc/` is generated.
`dist/sym.js` and `dist/sym.mjs` provide browser bundles. Set
`WLEARN_SYM_POLYGRAD_JS=/path/to/polygrad/js` for local Node checks and browser
builds. Run `npm test`, `npm run test:types`, `npm run test:polygrad`, and
`npm run test:browser`; WebGPU qualification uses `POLY_DEV=webgpu`.

See the [repository numerical contract](https://github.com/wlearn-org/sym#family-search-and-numerical-contract)
and LICENSE/NOTICE for semantics and licensing.

Runtime reuse, the AutoML factory recipe and remaining numerical/cache
limits are documented in the
[repository qualification](https://github.com/wlearn-org/sym#runtime-reuse-qualification).
Old runtimes fail early with a Polygrad 0.6 API requirement, before tensors are allocated.

Hierarchical division can produce very large conservative feature bounds even
from bounded input data. With Polygrad and trigonometric operators, this can
trigger the cosine precision guard during search. Use the default flat family
or the C backend for those searches; the error does not silently change backends.

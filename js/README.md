# @wlearn/sym

Browser and Node bindings for wlearn symbolic models backed by the C11/WASM core.

## Install

```bash
npm install @wlearn/sym @wlearn/core
```

Install `polygrad` only if you need optional finalist execution through `predictPolygrad()`, post-fit constant adaptation through `refinePolygrad()`, or the explicit `engine: "pg-family"` search path.

```bash
npm install polygrad@^0.6.0
```

## Regression

```js
const { SymbolicRegressor } = require('@wlearn/sym')

const X = [[0, 1], [1, 2], [2, 3]]
const y = [1, 3, 5]

const model = await SymbolicRegressor.create({
  population: 256,
  generations: 120,
  maxNodes: 31,
  operatorSet: 'full',
  seed: 42
})

model.fit(X, y)

const pred = model.predict([[3, 4]])
const text = model.formula({ format: 'text' })
const frontier = model.frontier()

const bytes = model.save('sym-reg.wlrn') // Node: writes the file and returns bytes
const restored = await SymbolicRegressor.load('sym-reg.wlrn')
const restoredPred = restored.predict([[3, 4]])
```

In browsers, use `model.save()` and store the returned bytes in IndexedDB, Cache Storage, or a download blob.

## Search strategy and backend

`strategy: 'tree', backend: 'c'` is the default. Regression and classification
also support `strategy: 'family', backend: 'c'`, with search and prediction owned
by C. Both C paths fit synchronously and need no Polygrad. FormulaTransformer is
tree-only. `backend: 'polygrad'` is not implemented for shared search yet;
`engine: 'pg-family'` retains the separate legacy algorithm. Conflicting settings
raise an error. Changing strategy/backend invalidates the fitted model.

Family controls include `terms` (default 6), `population` (128), `generations`
(20), `eliteCount` (8), `islands` (1), `immigrantRate` (0), `polishPasses` (0),
and `ridge` (1e-8). `operatorSet` accepts basic/smooth/full or an operator list.
Fitting uses double QR, an unpenalized intercept and MSE plus ridge coefficient
penalty. A seeded `validationFraction` holds rows out of coefficient fitting.
Classification preserves ±1 margin fitting and sigmoid/softmax probabilities;
it does not perform probability calibration. Tree-only and hierarchical controls
are rejected. The repository README specifies numerical and selection semantics.

New C-family fits use WLRN `wlearn.sym.regressor@2`/`wlearn.sym.classifier@2`
with a `SYM2` family payload and explicit protected-operator semantics. Unused
term parameters are canonicalized and excluded from complexity. Legacy `@1`
JSON remains readable and is re-saved without reinterpreting its metrics.
Tree bundles remain `@1`. Loading C-family artifacts requires no Polygrad.
Shared Polygrad scoring is not implemented yet. AutoML includes zero immigrants;
the C search exposes feature tiles while retaining QR and sequential polishing.
The static class typeId identifies the tree format; `save()` selects the family
`@2` typeId for newly fitted C-family models.

```js
const model = await SymbolicRegressor.create({
  strategy: 'family', backend: 'c', terms: 6, ridge: 1e-6, seed: 42
})
model.fit(X, y) // synchronous C search
```

## Classifier And Transformer

```js
const { SymbolicClassifier, FormulaTransformer } = require('@wlearn/sym')

const clf = await SymbolicClassifier.create({ seed: 7 })
clf.fit([[0], [1], [2], [3]], [10, 10, 20, 20])
clf.predict([[1.5]])
clf.predictProba([[1.5]])

const ft = await FormulaTransformer.create({ topK: 4, seed: 7 })
ft.fit([[0], [1], [2], [3]], [0, 1, 4, 9])
ft.transform([[4], [5]])
```

## API

- `await SymbolicRegressor.create(params)`
- `await SymbolicClassifier.create(params)`
- `await FormulaTransformer.create(params)`
- `fit(X, y)`
- `predict(X)`
- `score(X, y)`
- `predictProba(X)` on classifiers
- `decisionFunction(X)`
- `transform(X)` on `FormulaTransformer`
- `formula({ format: 'json' | 'text', index?: number })`
- `frontier()`
- `verify(checks)`
- `predictPolygrad(X, opts?)`
- `refinePolygrad(X, y, opts?)` on regressors
- `getParams()`, `setParams(params)`
- `save(path?)`
- `static load(bytesOrPath)`
- `dispose()`
- `static defaultSearchSpace()`

`create()` and `load()` are async because WASM instantiation is async. `fit`, `predict`, `score`, and `save` are synchronous after construction.

Exception: `engine: "pg-family"` uses the public Polygrad runtime during search, so `fit`, `predict`, `score`, and `predictProba` should be awaited for that engine.

Call `dispose()` in workers, cross-validation, and AutoML loops when many models are created and discarded.

## Parameters

- `population`
- `generations`
- `maxNodes`
- `maxDepth`
- `frontierSize`
- `topK`
- `operatorSet`: `basic`, `smooth`, `full`
- `loss`: `mse`, `mae`, `huber`
- `complexityPenalty`
- `mutationRate`
- `crossoverRate`
- `constantRate`
- `islands`
- `migrationInterval`
- `migrationCount`
- `warmupGenerations`
- `warmupMinNodes`
- `broodSize`
- `rowSampleSize`
- `localRefineInterval`
- `localRefineCount`
- `complexityHofSize`
- `complexityBucketWidth`
- `finalSelector`: `objective`, `loss`, `score`
- `seed`

PG-family-only:

- `engine`: set to `pg-family`
- `terms`
- `operators`
- `stackSummaries`

## Search Controls

The default path is the C search engine. Advanced controls are explicit and may be left unset:

- `islands`, `migrationInterval`, `migrationCount`: split the population into independent islands and periodically inject frontier formulas.
- `warmupGenerations`, `warmupMinNodes`: start with smaller formulas and linearly unlock `maxNodes`.
- `broodSize`: generate several children from the same parents and keep the best child.
- `rowSampleSize`: score offspring on a deterministic row subset, then rescore survivors/frontier formulas on the full data.
- `localRefineInterval`, `localRefineCount`: run C-side MSE coefficient refinement during search.
- `complexityHofSize`, `complexityBucketWidth`: keep the best formula per complexity bucket.
- `finalSelector`: choose the deployed formula by penalized objective, full-data loss, or complexity-archive score.

`frontier()` returns an objective-sorted archive. It is not guaranteed to be a strict nondominated Pareto front.

`defaultSearchSpace()` exposes only conservative values for AutoML. Larger values can improve difficult searches but increase fit time.

## Execution

Training uses the bundled C/WASM engine by default. The engine uses protected operators, deterministic seeds, tournament selection, crossover, mutation, C-side coefficient refinement for MSE finalists, affine output rescaling, and frontier tracking.

`predictPolygrad()` lowers selected fitted formulas to Polygrad Tensor execution. It is optional and not used by default `fit()` or `predict()`.

`refinePolygrad()` is an optional second refinement pass. It lowers a selected regression formula after C search, optimizes its constants with Polygrad, and commits the constants back into the C model only when training MSE improves or ties. The default Adam learning rate is conservative (`0.001`) because C-side coefficient refinement usually leaves little room for safe gradient updates. Saved bundles include committed constants. Unsupported formula graphs return an `unsupported` report instead of being treated as refined.

On a fitted C-engine regressor:

```js
const refinement = await model.refinePolygrad(X, y, { epochs: 80, lr: 0.001 })
```

`engine: "pg-family"` is an explicit search engine for regression and classification. It searches compact formula families with Polygrad summary kernels, then saves a normal `wlearn.sym.*@1` bundle with `metadata.engine = "pg-family"`. Python can load those bundles and can fit the same engine through `wlearn_sym`.

```js
const model = await SymbolicRegressor.create({
  engine: 'pg-family',
  operatorSet: 'basic',
  population: 128,
  generations: 20,
  terms: 6,
  seed: 42
})

await model.fit(X, y)
const pred = await model.predict(X)
const bytes = model.save('sym-pg.wlrn')
model.dispose()
```

By default, PG-family uses Polygrad's default runtime, matching ordinary
`const { Tensor } = require('polygrad')` usage. Pass `polygrad: pg` only when an
app wants an explicit runtime for device selection or isolation:

```js
const polygrad = require('polygrad')
const pg = polygrad.create({ core: 'wasm', device: 'webgpu' })
const model = await SymbolicRegressor.create({ engine: 'pg-family', polygrad: pg })
```

`polygrad: { core: 'wasm', device: 'auto' }` also remains accepted for an
isolated package-managed runtime.

PG-family operator control is portable and artifact-safe:

- `operatorSet`: `basic`, `smooth`, or `full`
- `operators`: explicit subset of built-ins such as `['add', 'sub', 'mul', 'sin']`
- `stackSummaries`: defaults to `auto`; JS uses one stacked summary tensor per coefficient solve
- `evalCacheSize`: descriptor-result cache size; set `0` to disable reuse of unchanged elites and repeated descriptors
- `scoreMode`: defaults to `summary`; use `cpu` or `polygrad` only when you need an explicit post-solve rescore

Arbitrary JavaScript functions are not accepted as operators because they cannot be serialized, verified, or executed in Python/browser bundles without a registered kernel ABI.

For local Polygrad development, point `WLEARN_SYM_POLYGRAD_JS` at that checkout's `js/` directory. Published packages use the npm `polygrad` peer.

`verify()` in JS proves only local protected-domain facts and reports unsupported exact checks honestly. Z3-backed checks are available in the Python package when `z3-solver` is installed.

## Local Polygrad 0.6 migration

Use the local 0.6.0 candidate until upstream release acceptance; the install
commands above apply after publication. See the [migration contract](../README.md#local-polygrad-06-migration)
for protected arithmetic, ownership, refinement acceptance and metric semantics.

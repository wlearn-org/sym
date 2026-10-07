# @wlearn/sym

Symbolic regression and classification for JavaScript: search for readable
formulas that fit your data. The search runs in WebAssembly in Node and
browsers. An optional [Polygrad](https://github.com/polygrad/polygrad) backend
scores candidates on CPU, CUDA or WebGPU. Part of
[wlearn](https://github.com/wlearn-org/wlearn).

## Install

```sh
npm install @wlearn/sym
npm install polygrad   # optional, only for the Polygrad backend (0.7 or newer)
```

## Quick start

```js
const { SymbolicRegressor } = require('@wlearn/sym')

async function main() {
  const X = [[0, 1], [1, 2], [2, 3], [3, 4]]
  const y = [1, 3, 5, 7]

  const model = await SymbolicRegressor.create({
    strategy: 'family', backend: 'c', terms: 3,
    population: 32, generations: 10, seed: 42
  })
  model.fit(X, y)
  console.log(model.predict([[4, 5]]))
  console.log(model.formula({ format: 'text' }))

  const restored = await SymbolicRegressor.load(model.save())
  console.log(restored.predict([[4, 5]]))
  restored.dispose()
  model.dispose()
}

main().catch(error => { console.error(error); process.exitCode = 1 })
```

## Models

- `SymbolicRegressor`: regression.
- `SymbolicClassifier`: binary and multiclass classification. Adds
  `predictProba`, `decisionFunction` and `classes`.
- `FormulaTransformer`: learns formula features to use as inputs for another
  model. Adds `transform` and `fitTransform`.

## Search strategies

| `strategy` + `backend` | What it searches | Models | `fit` |
| --- | --- | --- | --- |
| `tree` + `c` (default) | Expression trees, genetic programming | All three | Synchronous |
| `family` + `c` | Sums of up to `terms` nonlinear terms with a linear readout | Regressor, Classifier | Synchronous |
| `family` + `polygrad` | Same search; candidates scored with Polygrad | Regressor, Classifier | Promise |

`tree` + `polygrad` is not supported and throws an error. Awaiting `fit` is
always safe.

## Main parameters

Tree search (defaults): `population` (256), `generations` (120), `maxDepth` (6),
`operatorSet` (`'full'`), `complexityPenalty` (0.001), `seed` (42).

Family search (defaults): `terms` (6, from 1 to 32), `population` (128),
`generations` (20), `hierarchical` (false), `ridge` (1e-8), `polishPasses` (0),
`seed` (42).

`operatorSet` is `'basic'`, `'smooth'`, `'full'` or a list of operator names.
`validationFraction` holds out rows for model selection. All parameters are
listed in `index.d.ts`.

## API

- `static async create(params)` creates a model.
- `fit(X, y)`: synchronous for the C backend, a Promise for the Polygrad
  backend. `X` is an array of rows, a wlearn `DenseMatrix`, or a flat typed
  array.
- `predict(X)` and `score(X, y)` use the fitted formula; they never need
  Polygrad.
- `formula({ format: 'text' | 'json', index })` returns the fitted formula.
  `index` selects a class for multiclass models.
- `frontier()` returns the best formulas found, sorted by objective.
- `getParams()` and `setParams(params)` read and change parameters. Changing
  the strategy or backend clears the fitted model.
- `save()` returns the model as bytes; `save(path)` also writes a file
  (Node only). `static async load(bytes)` restores a model.
- `dispose()` releases resources. Call it when replacing models in
  long-running applications.
- `capabilities` and `static defaultSearchSpace()` support wlearn AutoML.

## Polygrad backend

```js
const { SymbolicRegressor } = require('@wlearn/sym')

async function main() {
  const X = Array.from({ length: 64 }, (_, i) => [i / 16, Math.sin(i / 5)])
  const y = X.map(([a, b]) => 2 * a + Math.sin(b))

  const model = await SymbolicRegressor.create({
    strategy: 'family', backend: 'polygrad', terms: 4, seed: 42,
    polygrad: { core: 'native', device: 'cpu' }   // or device: 'cuda'
  })
  await model.fit(X, y)
  const report = await model.refinePolygrad(X, y, { epochs: 20, lr: 0.001 })
  console.log(model.formula({ format: 'text' }), report)
  model.dispose()
}

main().catch(error => { console.error(error); process.exitCode = 1 })
```

- In browsers, use `polygrad: { core: 'wasm', device: 'webgpu' }`.
- To share a runtime between models, create it with `createAsync` from
  `polygrad/async` and pass it as `polygrad: runtime`. A shared runtime is
  never disposed by Sym.
- `refinePolygrad(X, y, options)` tunes the fitted formula's constants by
  gradient descent. Acceptance uses the C evaluator: family search uses its
  configured validation split; tree search compares mean squared error on the
  supplied rows, not the original held-out split. Pass the same rows used for `fit`.
- `predictPolygrad(X)` evaluates the fitted formula on Polygrad and returns a
  Promise.
- Kernel compilation has a setup cost, so small fits are usually faster with
  the default C backend.

## Saving and loading

`save()` uses the wlearn `.wlrn` format. The Python package `wlearn-sym` reads
the same files, and models saved by earlier versions still load. Loaders are
registered with `@wlearn/core`, so Sym models can be nested in pipelines and
ensembles.

## Notes and limitations

- Classifier probabilities come from sigmoid or softmax scores and are not
  calibrated.
- Results are reproducible for a seed on the same backend. Polygrad uses
  float32 on WebGPU and float64 on native devices, which can pick a different
  formula when candidates are nearly tied.
- With the Polygrad backend, hierarchical families that combine division with
  trigonometric operators can exceed a precision guard and raise an error. Use
  the flat family or the C backend for those searches.
- Experimental controls (`scaleAware`, `lossScale`, `polishMethod: 'lm'`,
  `localRefineInterval`) are described in the
  [repository README](https://github.com/wlearn-org/sym#family-search-and-numerical-contract).

## Compatibility

The optional Polygrad backend requires `polygrad` 0.7 or newer. Uses
`@wlearn/core` 0.3.

## License

Apache-2.0. See `NOTICE` and `licenses/` for third-party attributions.

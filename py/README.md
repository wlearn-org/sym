# wlearn-sym

Symbolic regression and classification for Python: search for readable
formulas that fit your data. The search runs in a C extension. An optional
[Polygrad](https://github.com/polygrad/polygrad) backend scores candidates on
CPU or CUDA. Part of [wlearn](https://github.com/wlearn-org/wlearn).

## Install

```sh
pip install wlearn-sym
pip install 'wlearn-sym[polygrad]'   # optional Polygrad backend
pip install 'wlearn-sym[verify]'     # optional Z3 formula checks
```

Requires Python 3.10 or newer. The package is distributed as source and
compiles a C extension during installation, so a C compiler is needed.

## Quick start

```python
from wlearn_sym import SymbolicRegressor

X = [[0, 1], [1, 2], [2, 3], [3, 4]]
y = [1, 3, 5, 7]

model = SymbolicRegressor({
    "strategy": "family", "backend": "c", "terms": 3,
    "population": 32, "generations": 10, "seed": 42,
}).fit(X, y)
print(model.predict([[4, 5]]))
print(model.formula(format="text"))

restored = SymbolicRegressor.load(model.save())
print(restored.predict([[4, 5]]))
restored.dispose()
model.dispose()
```

## Models

- `SymbolicRegressor`: regression.
- `SymbolicClassifier`: binary and multiclass classification. Adds
  `predict_proba`, `decision_function` and `classes`.
- `FormulaTransformer`: learns formula features to use as inputs for another
  model. Adds `transform` and `fit_transform`.

Parameters are passed as a dict. Both camelCase and snake_case names are
accepted.

## Search strategies

| `strategy` + `backend` | What it searches | Models |
| --- | --- | --- |
| `tree` + `c` (default) | Expression trees, genetic programming | All three |
| `family` + `c` | Sums of up to `terms` nonlinear terms with a linear readout | Regressor, Classifier |
| `family` + `polygrad` | Same search; candidates scored with Polygrad | Regressor, Classifier |

`tree` + `polygrad` is not supported and raises an error. All methods are
synchronous.

## Main parameters

Tree search (defaults): `population` (256), `generations` (120), `maxDepth` (6),
`operatorSet` (`'full'`), `complexityPenalty` (0.001), `seed` (42).

Family search (defaults): `terms` (6, from 1 to 32), `population` (128),
`generations` (20), `hierarchical` (False), `ridge` (1e-8), `polishPasses` (0),
`seed` (42).

`operatorSet` is `'basic'`, `'smooth'`, `'full'` or a list of operator names.
`validationFraction` holds out rows for model selection.

## API

- `fit(X, y)` returns the model. `X` is a NumPy array or a list of rows.
- `predict(X)` and `score(X, y)` use the fitted formula; they never need
  Polygrad.
- `formula(format='text' | 'json', index=0)` returns the fitted formula.
  `index` selects a class for multiclass models.
- `frontier()` returns the best formulas found, sorted by objective.
- `verify(checks=None)` checks tree formulas with Z3 (needs the `verify` extra).
- `get_params()` and `set_params(**params)` read and change parameters.
  Changing the strategy or backend clears the fitted model.
- `save(path=None)` returns the model as bytes and optionally writes them.
  `load(source)` accepts bytes or a path.
- `dispose()` releases resources. Call it when replacing models in
  long-running loops.
- `capabilities` and `default_search_space()` support wlearn AutoML.

## Polygrad backend

```python
import numpy as np
from wlearn_sym import SymbolicRegressor

X = np.column_stack([np.arange(64) / 16, np.sin(np.arange(64) / 5)])
y = 2 * X[:, 0] + np.sin(X[:, 1])

model = SymbolicRegressor({
    "strategy": "family", "backend": "polygrad", "terms": 4, "seed": 42,
    "polygrad": {"device": "CPU"},   # or {"device": "CUDA"}
}).fit(X, y)
report = model.refine_polygrad(X, y, epochs=20, lr=0.001)
print(model.formula(format="text"), report)
model.dispose()
```

- `polygrad` is a dict of `polygrad.create()` options or an existing runtime.
  A runtime you pass is never disposed by Sym and keeps compiled kernels
  between fits.
- `refine_polygrad(X, y, epochs, lr)` tunes the fitted formula's constants by
  gradient descent. Acceptance uses the C evaluator: family search uses its
  configured validation split; tree search compares mean squared error on the
  supplied rows, not the original held-out split. Pass the same rows used for `fit`.
- `predict_polygrad(X, polygrad=None)` evaluates the fitted formula on Polygrad.
- Kernel compilation has a setup cost, so small fits are usually faster with
  the default C backend.

## Saving and loading

`save()` uses the wlearn `.wlrn` format. The npm package `@wlearn/sym` reads the
same files, and models saved by earlier versions still load. Loaders are
registered with `wlearn`, so Sym models can be nested in pipelines and
ensembles.

## Notes and limitations

- Classifier probabilities come from sigmoid or softmax scores and are not
  calibrated.
- Results are reproducible for a seed on the same backend. Different devices
  and precisions can pick a different formula when candidates are nearly tied.
- With the Polygrad backend, hierarchical families that combine division with
  trigonometric operators can exceed a precision guard and raise an error. Use
  the flat family or the C backend for those searches.
- Experimental controls (`scaleAware`, `lossScale`, `polishMethod: 'lm'`,
  `localRefineInterval`) are described in the
  [repository README](https://github.com/wlearn-org/sym#family-search-and-numerical-contract).

## Compatibility

The optional Polygrad backend requires `polygrad` 0.7 or newer. Requires
`wlearn` 0.3.

## License

Apache-2.0. See `NOTICE` for third-party attributions.

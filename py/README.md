# wlearn-sym

Python bindings for wlearn symbolic models backed by the C11 native core.

## Install

```bash
pip install wlearn-sym wlearn
```

Optional extras:

```bash
pip install "wlearn-sym[polygrad]"  # polygrad>=0.6.0,<0.7; predict_polygrad(), refine_polygrad(), pg-family
pip install "wlearn-sym[verify]"    # Z3-backed verification
```

`import wlearn_sym` does not import Polygrad or Z3. Optional dependencies are imported only when their methods are called.

## Regression

```python
from wlearn_sym import SymbolicRegressor

X = [[0, 1], [1, 2], [2, 3]]
y = [1, 3, 5]

model = SymbolicRegressor({
    "population": 256,
    "generations": 120,
    "maxNodes": 31,
    "operatorSet": "full",
    "seed": 42,
})

model.fit(X, y)

pred = model.predict([[3, 4]])
text = model.formula(format="text")
frontier = model.frontier()

bytes_ = model.save("sym-reg.wlrn")
restored = SymbolicRegressor.load("sym-reg.wlrn")
restored.predict([[3, 4]])
```

`save()` returns `.wlrn` bytes. `save(path)` writes the same bytes and returns them. `load()` accepts bytes, `str`, or `PathLike`.

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

```python
model = SymbolicRegressor({
    "strategy": "family", "backend": "c", "terms": 6, "ridge": 1e-6, "seed": 42,
}).fit(X, y)
```

## Classifier And Transformer

```python
from wlearn_sym import FormulaTransformer, SymbolicClassifier

clf = SymbolicClassifier({"seed": 7})
clf.fit([[0], [1], [2], [3]], [10, 10, 20, 20])
clf.predict([[1.5]])
clf.predict_proba([[1.5]])

ft = FormulaTransformer({"topK": 4, "seed": 7})
ft.fit([[0], [1], [2], [3]], [0, 1, 4, 9])
ft.transform([[4], [5]])
```

## API

- `SymbolicRegressor(params=None)`
- `SymbolicClassifier(params=None)`
- `FormulaTransformer(params=None)`
- `fit(X, y)`
- `predict(X)`
- `score(X, y)`
- `predict_proba(X)` on classifiers
- `decision_function(X)`
- `transform(X)` on `FormulaTransformer`
- `formula(format="json", index=0)`
- `frontier()`
- `verify(checks=None)`
- `predict_polygrad(X, polygrad=None)`
- `refine_polygrad(X, y, index=0, epochs=80, lr=0.001, optimizer="adam", polygrad=None)` on regressors
- `get_params()`, `set_params(params=None, **kwargs)`
- `save(path=None)`
- `load(bytes_or_path)`
- `dispose()`
- `default_search_space()`

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

- `engine`: set to `"pg-family"`
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

`default_search_space()` exposes only conservative values for AutoML. Larger values can improve difficult searches but increase fit time.

## Execution

Training uses the compiled C core. The core owns formula generation, protected evaluation, deterministic seed formulas, tournament selection, mutation, crossover, C-side coefficient refinement for MSE finalists, affine MSE rescaling, and raw `SYM1` serialization.

Python also supports explicit `engine: "pg-family"` for Polygrad formula-family search:

```python
model = SymbolicRegressor({
    "engine": "pg-family",
    "operatorSet": "basic",
    "population": 128,
    "generations": 20,
    "terms": 6,
    "seed": 42,
})
model.fit(X, y)
pred = model.predict(X)
model.dispose()
```

By default, PG-family uses Polygrad's module-level runtime, matching ordinary
`from polygrad import Tensor` usage. Pass `{"polygrad": pg}` only when an app
wants an explicit runtime/context, or `{"polygrad": {"device": "cpu"}}` for an
isolated package-managed runtime. Runtime selection is execution wiring and is
not stored in `.wlrn` artifacts.

PG-family bundles use the same `wlearn.sym.*@1` type IDs as JS and set `metadata.engine = "pg-family"`.

PG-family operator control is portable and artifact-safe:

- `operatorSet`: `basic`, `smooth`, or `full`
- `operators`: explicit subset of built-ins such as `["add", "sub", "mul", "sin"]`
- `stackSummaries`: defaults to `"auto"`; Python uses Polygrad's batched `numpy_many` readback unless explicitly set to `True`
- `evalCacheSize`: descriptor-result cache size; set `0` to disable reuse of unchanged elites and repeated descriptors
- `scoreMode`: defaults to `"summary"`; use `"cpu"` only when you need an explicit post-solve rescore

Arbitrary Python callables are not accepted as operators because they cannot be serialized, verified, or executed in JS/browser bundles without a registered kernel ABI.

`predict_polygrad()` is optional and imports Polygrad only when called. Pass
`polygrad=pg` to use a caller-owned runtime. `refine_polygrad()` is an optional
second refinement pass: it lowers one fitted regression formula after C search,
optimizes constants with Polygrad, and commits constants back into the native
model only when training MSE improves or ties. The default Adam learning rate is
conservative (`0.001`) because C-side coefficient refinement usually leaves
little room for safe gradient updates. Saved bundles include committed constants.
Unsupported formula graphs return an `unsupported` report.

On a fitted C-engine regressor:

```python
refinement = model.refine_polygrad(X, y, epochs=80, lr=0.001)
```

`verify()` uses Z3 for supported monotonicity, equivalence, and range checks when `z3-solver` is installed; unsupported checks return `unknown` or `unsupported`.

Call `dispose()` in long-running loops that create many native model instances.

## Local Polygrad 0.6 migration

Use the local 0.6.0 candidate until upstream release acceptance; the install
commands above apply after publication. See the [migration contract](../README.md#local-polygrad-06-migration)
for protected arithmetic, ownership, refinement acceptance and metric semantics.

# wlearn-sym

Python bindings for symbolic regression, classification and formula features.
The C11 core owns search and portable fitted models. Optional Polygrad execution
uses its public Python frontend; importing `wlearn_sym` does not import Polygrad.

The optional Polygrad backend uses version **0.6.0**.

```sh
pip install wlearn-sym
pip install 'wlearn-sym[polygrad]'  # optional scoring and gradient refinement
pip install 'wlearn-sym[verify]'    # optional Z3 tree verification
```

```python
from wlearn_sym import SymbolicRegressor
X = [[0, 1], [1, 2], [2, 3], [3, 4]]
y = [1, 3, 5, 7]
model = SymbolicRegressor({
    "strategy": "family", "backend": "c", "terms": 3,
    "population": 32, "generations": 10, "seed": 42,
}).fit(X, y)
predictions = model.predict([[4, 5]])
text = model.formula(format="text")
bytes_ = model.save()
restored = SymbolicRegressor.load(bytes_)
```

## API and execution

`SymbolicRegressor`, `SymbolicClassifier` and `FormulaTransformer` expose
`fit`, `predict`, `score`, `get_params`, `set_params`, `save`, `load`, `dispose`,
`capabilities` and `default_search_space`. Classifiers add `predict_proba`,
`decision_function` and `classes`; FormulaTransformer adds `transform` and
`fit_transform`. Methods are synchronous. Inputs accept NumPy matrices and lists.

`formula(format='json'|'text', index=0)` exposes expressions; `frontier()` returns
an objective-sorted archive. `verify()` uses optional Z3 for supported tree checks
and reports unsupported family proofs explicitly.

`save()` returns WLRN bytes. `save(path)` writes and returns the same bytes;
`load` accepts bytes, `str`, or `PathLike`. Loaders register with `wlearn` for
nested Pipeline/ensemble persistence. Default prediction and loading use C,
including when training used Polygrad.

Default search is `strategy: 'tree', backend: 'c'`, supporting all three model
classes. Family search supports regression/classification with either `c` or
`polygrad`. Tree+Polygrad search and family Transformer raise errors.

```python
accelerated = SymbolicRegressor({
    "strategy": "family", "backend": "polygrad", "seed": 42,
    "validationFraction": .2, "hierarchical": False,
    "polishPasses": 2, "polishBatchSize": 32,
    "polygrad": {"device": "CUDA"},
}).fit(X, y)
report = accelerated.refine_polygrad(X, y, epochs=20, lr=.001)
```

`polygrad: runtime` borrows a caller-created runtime; Sym never disposes it.
Without one, Sym creates and disposes its own runtime. `predict_polygrad(X,
polygrad=runtime)` explicitly executes the fitted formula on Polygrad.
Call model `dispose()` when replacing models in long-running loops.

Family controls: `terms` 1–32 (default 6), `population` (128), `generations` (20),
`eliteCount` (8), `islands` (1), `immigrantRate` (0), `ridge` (1e-8),
`polishPasses` (0), `polishBatchSize` (0), `hierarchical` (False). Positive polish
batch sizes freeze a base and score independent parameter variants. Zero keeps
sequential coordinate polish. Hierarchical terms may reference earlier terms.
`operatorSet` accepts basic/smooth/full or a list; `operators` selects portable
built-ins. Arbitrary Python callbacks are not artifact operators.

The shared C controller proposes candidates and accepts results. Polygrad
computes features, centered/scaled QR factors and losses; C applies ridge and
solves small triangular systems. `batchSize` (default `min(population, 512)`) controls transport,
`scorerDtype` defaults to float64. Precision/reduction differences can affect
near-tied selections. Unsupported large cosine phases raise explicit errors;
no scorer silently falls back. Small fits generally favor C because Polygrad
compilation has a setup cost.

Family readouts fit MSE plus ridge, including ±1 classification margins.
Sigmoid/softmax probabilities are not calibrated. Readouts and optional gradient
refinement exclude the deterministic validation split. C accepts gradient
updates only when validation-plus-complexity objective does not worsen beyond
tolerance. Reuse the same training rows/order; `index` selects a multiclass
head. Structure stays fixed. Tree refinement retains regression-only,
training-MSE acceptance on the supplied data.

`engine: 'pg-family'` now aliases shared C search with Polygrad scoring. The old
frontend search was removed, so refitting can differ. Prefer strategy/backend;
legacy scorer controls such as `stackSummaries`/`scoreMode` are unsupported.

## Persistence and development

Tree WLRN typeIds use `@1`; flat families `@2`; hierarchical families `@3`.
Legacy family JSON `@1` loads without Polygrad and keeps historical metrics when
re-saved. Refitting writes the current format. The native payload validates
source indices, dimensions and version/typeId correspondence.

The repository `src/` owns C source; `py/csrc/` is generated for builds. For local
qualification set `SYM_LIB_PATH` to the freshly built Sym library, `POLY_LIB` to
Polygrad's library, and include the respective `py/` directories on `PYTHONPATH`.
Run `python -m pytest py/tests test/test_cross_lang.py` from the repository.

See the [repository numerical contract](https://github.com/wlearn-org/sym#family-search-and-numerical-contract)
and LICENSE/NOTICE for semantics and licensing.

A caller-owned `polygrad.create(...)` runtime can be passed as `polygrad` across
fits. It reuses compiled kernels; dispose it after the bounded workload. See the repository runtime-reuse qualification for schedule-cache retention
and large-phase cosine limits; float64 is not a universal exact-match guarantee.

Hierarchical division can produce very large conservative feature bounds even
from bounded input data. With Polygrad and trigonometric operators, this can
trigger the cosine precision guard during search. Use the default flat family
or the C backend for those searches; the error does not silently change backends.


### In-loop family polish

Optional `localRefineInterval` and `localRefineCount` run family polish before
breeding every interval. Both default to zero; enable both with `polishPasses > 0`.
Count is at most `min(population, 32)`. Final polish still runs once; the last
generation does not also receive in-loop polish. Both scorers use the same C
proposal schedule and QR readout refits. This is an experimental coordinate
polish schedule, not an established general quality improvement.

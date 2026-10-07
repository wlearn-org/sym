# wlearn/sym

Symbolic regression, classification and supervised formula features. One C11
core owns search, candidate selection, fitted formulas and portable prediction.
JavaScript and Python provide wlearn lifecycle/artifacts and optional execution
through Polygrad's public frontends. No private CUDA API or direct link to
Polygrad's C core is required.

Optional accelerated execution requires published Polygrad **>=0.7.0**; tested with 0.7.0.
The default C backend does not require Polygrad.

## Strategies and backends

| Selection | Search | JavaScript `fit` |
| --- | --- | --- |
| `tree` + `c` | Default tree search | Synchronous |
| `family` + `c` | Additive or hierarchical family search | Synchronous |
| `family` + `polygrad` | Same C search, device feature evaluation/QR/loss | Promise |
| `tree` + `polygrad` | Unsupported; raises an error | - |

Both family backends support regression and binary/multiclass classification.
`FormulaTransformer` uses tree search. `predict`, `predictProba`, `score`, `save`
and `dispose` use the fitted C model synchronously, including after Polygrad
training. `predictPolygrad`/`predict_polygrad` explicitly execute the selected
formula on Polygrad. Python methods are synchronous.

The historical `engine: 'pg-family'` setting now aliases family+Polygrad. Its
former JS/Python search controllers have been removed; refitting can produce a
different formula. Old `engine: 'c'`, `'wasm'`, `'c-wasm'` and `'auto'` select
tree+C. Conflicting settings raise errors. Changing strategy/backend invalidates
the fitted model. Failed refits preserve the previous fitted model.

```js
const { SymbolicRegressor } = require('@wlearn/sym')
const model = await SymbolicRegressor.create({
  strategy: 'family', backend: 'c',
  terms: 6, population: 128, generations: 20, seed: 42
})
await model.fit(X, y)
const predictions = model.predict(Xtest)
const bytes = model.save()
```

```python
from wlearn_sym import SymbolicRegressor
model = SymbolicRegressor({
    "strategy": "family", "backend": "c", "seed": 42,
}).fit(X, y)
predictions = model.predict(Xtest)
bytes_ = model.save()
```

For accelerated training, install Polygrad and set `backend: 'polygrad'` with
`polygrad: { core: 'native', device: 'cuda' }` (Python: `{'device': 'CUDA'}`). For browser WebGPU, select
`polygrad: { core: 'wasm', device: 'webgpu' }`; Sym loads Polygrad's public async
frontend. Caller-supplied runtimes (`polygrad: runtime` or `polygradRuntime`) are
borrowed, including runtimes shared with another wlearn model. Sym disposes its
own stages and temporary Models, and only disposes runtimes it creates.
When no device is specified, Polygrad resolves `POLY_DEV` then `DEV` in native
processes. Browser code passes runtime options; the browser test harness forwards
those standard environment selectors into the page.

## Family search and numerical contract

Controls: `terms` 1 to 32 (default 6), `population` (128), `generations` (20),
`eliteCount` (8, less than population), `islands` (1), `immigrantRate` (0),
`polishPasses` (0), `polishBatchSize` (0), `hierarchical` (false), `ridge` (1e-8).
Each island selects within its own population. There is no family island
migration. `mutationRate` controls structural replacement and every child gets
annealed parameter jitter. Unchanged elites retain their readouts and losses.

The formula is a bias plus weighted terms. `operatorSet` accepts `basic`,
`smooth`, `full`, or a list; `operators` accepts names or IDs. Built-ins are
add/subtract/multiply/divide, sin/cos/tanh, logabs/sqrtabs/expclamp. Family division
and log use epsilon 1e-6; exp clamps its argument to +/-6. Tree operators retain
their distinct protection rules. Arbitrary language callables are not portable
operators.

With `hierarchical: true`, source indices below `nFeatures` select input columns;
`nFeatures + t` selects an earlier term `t`. Self/forward references are invalid.
The model stores this directed structure. Formula text uses shared `t0`, `t1`, ...
definitions so nested expressions cannot produce exponentially large strings.

Sequential polish (`polishBatchSize: 0`) evaluates dependent coordinate trials.
Positive sizes request frozen-base batches: all active coordinate directions
plus random multi-parameter proposals, followed by selection. The actual round
size is at least the coordinate-trial count. This changes the polish policy;
transport `batchSize` (1 to 512, default `min(population, 512)`) does not change proposal/RNG order.

For family regression, `lossScale: 'target-variance'` interprets
`complexityPenalty` and `tol` relative to the population variance of the rounded
training targets. The default is `'absolute'`. Validation rows never enter this
calculation; constant training targets give zero penalty and tolerance. MSE and
saved formulas retain original units, and the artifact stores the resolved
penalty. Ridge and the separate post-fit refinement `tolerance` stay unchanged.
This controls target-unit sensitivity; it does not guarantee an identical search
under floating-point rounding. Classification rejects this mode.

Optional `localRefineInterval` and `localRefineCount` enable family polish during
evolution. Set both positive, with `polishPasses > 0`; count is at most
`min(population, 32)`. Every interval, before breeding, up to that many distinct
best population candidates receive the same coordinate/batched polish and QR
readout refits. Their accepted updates enter the next generation. Final polish
still runs once; the last generation does not also receive in-loop polish.
Both settings default to zero. The default coordinate method is derivative-free;
quality gains are not established across datasets. C owns the schedule
and both C and Polygrad scorers execute its proposals.

Two further experimental controls are available for family search:

- `scaleAware: true` learns input means/scales from training rows and searches
  dimensionless nonlinear slopes/phases, with a mix of near-zero and signed-log
  initial slopes. Saved formulas stay in original input units. This does not
  scale targets or change the absolute complexity penalty.
- `polishMethod: 'lm'` uses finite differences of QR-refitted residuals, then a
  damped least-squares step. Set `polishPasses > 0` and leave `polishBatchSize` at
  zero. It works for final and in-search polish. Steps must improve regularized
  training loss and the existing validation objective before acceptance.

Both are opt-in. LM differentiates the rounded readout numerically; it is not an
analytic gradient; float32 residual differences can change its steps and search path. Each round adds probe fits and a residual workspace of up to
`(2 * terms + 1) * (rows + terms)` doubles. Polygrad still evaluates features, QR factors, losses and residuals on the
device; C owns the small coefficient and damped solves. The host receives
training residuals for the latter. GPU reductions for that system are future work.
Float32 storage and existing phase guards remain in force; large offsets may
still require input preprocessing. See [benchmarks](bench/README.md).

C fits readouts using centered/scaled double Givens QR for
`mean_squared_error + ridge * sum(coef^2)`, with an unpenalized intercept.
Rank-deficient pivots receive zero coefficients; positive ridge is advised.
Inputs, targets, term parameters, coefficients and returned C margins are
float32-rounded. Intermediate features, intercepts and C QR use double precision.

The Polygrad scorer keeps data and feature matrices on the device, computes
centered/scaled QR factors and explicit prediction losses, and transfers only
compact factors/statistics/readouts. C applies original-unit ridge and solves
the small triangular systems. This avoids raw-moment Gram cancellation. It does
not mean every operation is on the GPU. All proposed candidates, including
polish trials, use the selected scorer; there is no hidden C pre-ranking.

Native CPU/CUDA scoring defaults to float64. Browser WebGPU uses float32;
`scorerDtype` can explicitly select a supported precision. Different reductions
and precision can change near-tied search decisions. Large cosine phases that
exceed the scorer's precision budget raise an error with a C-backend suggestion;
there is no silent fallback. Tests compare predictions and losses with declared
tolerances, not identical search histories for arbitrary datasets/devices.

`validationFraction` creates a seeded holdout excluded from centering/readout
fitting. Selection, polish and early stopping use holdout MSE plus
`complexityPenalty * complexity`; without a holdout, they use training MSE.
The selected readout is not refit on the holdout. Classification fits +/-1 margins
and converts them with sigmoid/softmax; these probabilities are not calibrated.

`refinePolygrad(X, y, { epochs, lr, index })` / `refine_polygrad` use public
Polygrad Model autodiff to optimize active nonlinear parameters, readout weights
and intercept of a fixed family structure. Only training rows from the same
seeded split enter the optimizer. C recomputes losses and commits only when the
validation-plus-complexity objective does not worsen beyond `tolerance` (1e-10).
`index` selects a multiclass head; regression/binary use zero. Supply the same
training rows/order to reuse the search split. This holdout is a selection set,
not an unbiased final test set. Refinement is explicit, not an automatic fit step.

Tree refinement retains its existing regression-only behavior and training-MSE
acceptance on the supplied data. `FormulaVerifier` reports unsupported family
proofs rather than treating family terms as tree nodes. Python optionally uses
Z3 for supported tree checks.

## Artifacts and integration

Only WLRN bundles are public persistence. `save()` returns bytes; `save(path)`
writes those bytes in Python/Node. `load` accepts bytes or a path.

- Tree: `SYM1`, `wlearn.sym.{regressor,classifier,transformer}@1`.
- Flat family: `SYM2` semantics 2, `wlearn.sym.{regressor,classifier}@2`.
- Hierarchical family: `SYM2` semantics 3, `wlearn.sym.{regressor,classifier}@3`.
- Legacy family JSON `@1`: readable without Polygrad, with original metrics;
  re-saving preserves its legacy format. A refit writes the current format.

Static class `typeId` identifies the tree format; `save` selects the actual
format. Readers reject typeId/payload mismatches and invalid references.
Unused arithmetic parameters are fixed to `(1, 0)` and unary second sources to
zero. Complexity and archive identity ignore unused genes; general algebraic
equivalence is not deduplicated.

Both languages use the wlearn core bundle/registry and Pipeline contracts.
JS family+Polygrad `fit` is the documented asynchronous base-estimator exception;
Pipeline/AutoML propagate its Promise. Default AutoML search spaces use C and
include immigrant rate zero. Backend acceleration, hierarchy and polish are
explicit choices, not automatic quality improvements. Dispose models when
replacing them in long-running loops.

## Native interfaces and layout

`src/` is canonical C. `js/csrc/` and `py/csrc/` are ignored generated package
copies. `js/` publishes `@wlearn/sym`; `py/` publishes `wlearn-sym`.
`test/` includes legacy fixtures, C/JS/browser and cross-language checks.

Tree stepping is in `src/sym.h`. Family stepping is in `src/sym_family.h`:
propose/score/accept/finish/free plus a compact QR-results protocol. Candidate
batches are bounded at 512, row tiles at 4096. Invalid acceptance is atomic;
borrowed views must be copied before awaiting or growing Wasm memory. Finishing
transfers the model once; freeing a search cancels it. Sequential and batched
polish use the same C state machine. These low-level APIs are intended for custom scoring integrations.

The C fit driver's feature cache is capped at 8 MiB, using 8-candidate chunks
and a bounded row tile. Uncached suffixes are recomputed; cache allocation
failure falls back to uncached evaluation without changing fitted bytes.

## Build, test and benchmark

From this repository:

```sh
make test-c
npm --prefix js run build
npm --prefix js run build:browser
make test
make test-browser
make wheel npm-pack
```

Local Polygrad: set `WLEARN_SYM_POLYGRAD_JS=/path/to/polygrad/js` for Node/browser
builds, `WLEARN_POLYGRAD_PY=/path/to/polygrad/py` for Make, and `POLY_LIB` for its
fresh native library. For direct Python commands include `py`, wlearn's `py`,
and Polygrad's `py` on `PYTHONPATH`; set `SYM_LIB_PATH=$PWD/build/libsym.so`.

```sh
npm --prefix js run test:polygrad
POLY_DEV=webgpu node test/test-browser.js
sh bench/scorer/build.sh
python -m pytest bench/scorer/test_scorer.py
python bench/family-backends.py --rows 4096 --population 256 --generations 120 --terms 8 --output results.json
```

WebGPU checks need a GPU-enabled full Chromium and a display. Tests fail with
the actual backend error rather than counting an unavailable backend as passed.
Keep CPU concurrency below half the machine's cores.

`bench/family-backends.py` measures complete fits, including runtime creation
and graph compilation, on fresh estimators with matched seeds and held-out test
rows. `bench/scorer/` measures fixed-candidate numerical/steady-state behavior;
its timings exclude parts of estimator fitting and are not end-to-end speedups.
Historical experiments under sibling `symcpg` and `sympg` are reference evidence,
not product benchmark results. C remains the default, especially for small fits.

## Runtime reuse qualification

Pass a caller-owned Polygrad runtime to reuse its compiled-kernel cache across
fits; dispose that runtime at the end of the bounded workload. Current scorers
are rebuilt for each fit. A device/dtype/shape-only scorer cache is insufficient:
normalization graphs also capture the target mean, training count and anchor row.

AutoML candidate parameters must remain portable JSON. Inject the runtime through
the estimator factory, outside the candidate parameters:

```js
const pg = require('polygrad/async')
const { autoFit } = require('@wlearn/automl')
const runtime = await pg.createAsync({ core: 'native', device: 'cuda' })
class SharedSym extends SymbolicRegressor {
  static create(params) {
    return SymbolicRegressor.create({ ...params, polygrad: runtime })
  }
}
let result
try {
  result = await autoFit([{
    name: 'sym', cls: SharedSym, searchSpace: {},
    params: { strategy: 'family', backend: 'polygrad', terms: 4,
      population: 32, generations: 10 }
  }], X, y, { nIter: 1, cv: 2, task: 'regression', ensemble: false })
  const predictions = result.model.predict(Xtest)
} finally {
  result?.model.dispose()
  runtime.dispose()
}
```

Polygrad 0.6.0 fixes the earlier repeated-fit graph retention and CPU float64
sine failure; the offset regression passes on CPU and CUDA. Runtime schedule
caches can still retain device storage, so keep their lifetimes bounded.
Large-phase cosine and gradient accuracy remain limitations of the optional
backend. Saved models predict with the C evaluator. Float64 is not a universal
exact-match guarantee; see `bench/README.md` for the measured precision limits.

Hierarchical division can produce very large conservative feature bounds even
from bounded input data. With Polygrad and trigonometric operators, this can
trigger the cosine precision guard during search. Use the default flat family
or the C backend for those searches; the error does not silently change backends.

## Release history

### 0.2.0 (2026-10-02)

- Add optional in-loop family polish, train-only scaled constant coordinates,
  and damped least-squares refinement through C and public Polygrad scorers.
- Add optional regression loss scaling by training-target variance. Existing
  defaults, artifact formats and the public C parameter struct remain compatible.
- Add reproducible quality/cost benchmarks, numerical and cross-runtime tests,
  and preserve large starting constants during post-fit gradient refinement.

These controls are experimental and opt-in. Polygrad 0.7.0 is the tested runtime.

## License

Apache-2.0. See [js/LICENSE](js/LICENSE) and [js/NOTICE](js/NOTICE).

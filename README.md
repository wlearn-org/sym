# wlearn/sym

Symbolic regression, symbolic classification, and supervised formula features for wlearn.

`sym` is a C-first original wlearn model package. The C11 core owns tree and additive-family search, protected operator evaluation, frontier tracking and prediction. Tree models use raw `SYM1` serialization; new family models use a versioned `SYM2` family payload inside WLRN; legacy JSON remains readable. JS and Python wrap that core, package `.wlrn` bundles, and register loaders with wlearn. Polygrad is optional: the default path uses it only for finalist formula execution/refinement, while `engine: "pg-family"` exposes explicit Polygrad family search. Z3 is optional for post-fit verification in Python.

## Search strategy and backend

The default is `strategy: 'tree', backend: 'c'`. Regression and classification
also support `strategy: 'family', backend: 'c'`: C owns the population, RNG,
selection, coefficient fitting, archive and prediction. No Polygrad installation
is needed for either C path. `FormulaTransformer` remains tree-only.

| Selection | Status | JavaScript fit |
| --- | --- | --- |
| `tree` + `c` | Default tree search | Synchronous |
| `family` + `c` | Fixed-width additive family search | Synchronous |
| `family` + `polygrad` | Shared-search scorer not implemented | Rejected |
| `tree` + `polygrad` | Not implemented | Rejected |
| `engine: 'pg-family'` | Temporary legacy frontend search | Promise |

The old `engine: 'c'`, `'wasm'`, `'c-wasm'` and `'auto'` aliases resolve to tree+C.
Conflicting engine/strategy/backend settings raise an error. The legacy
`pg-family` engine remains a separate algorithm; selecting it explicitly may
also specify `strategy: 'family', backend: 'polygrad'`. It will be retired after
shared scoring and artifact conversion are qualified. `getParams()` records
resolved settings; changing strategy/backend with `setParams()` invalidates the
fitted model. A rejected configuration leaves the previous model intact.

### C family controls and numerical contract

Use `terms` (1–32, default 6), `population` (default 128), `generations` (20),
`eliteCount` (8, less than population), `islands` (1), `immigrantRate` (0),
`polishPasses` (0), and `ridge` (1e-8). Each island selects from its own top half;
elites are apportioned across islands, capped at half the island population.
Immigrants replace the configured fraction of each island, rounded down.
`mutationRate` controls feature/operator replacement; every mutated child also
receives annealed parameter jitter. There is no cross-island migration yet.

`operatorSet` accepts `basic`, `smooth`, `full`, or an explicit operator list;
`operators` also accepts a list. The grammar is a bias plus weighted terms drawn
from add, subtract, multiply, divide, sin, cos, tanh, logabs, sqrtabs and expclamp.
It uses division/log epsilon 1e-6 and an exponential clamp of ±6. These differ
from tree operators. Hierarchical earlier-term references are not supported yet.

Coefficient fitting uses centered/scaled double-precision Givens QR for mean
squared error plus `ridge * sum(coef²)`, with an unpenalized intercept. Zero or
very small QR pivots are assigned zero coefficients; positive ridge is advised
for rank-deficient features. Inputs, targets, term parameters, coefficients and
final predictions retain the existing family payload's float32 precision.
Intermediate operator evaluation, the intercept and QR arithmetic use doubles.

The C fit driver caches features across the moments, QR and loss passes for its
8-candidate chunks. The cache is capped at 8 MiB (3 MiB for 4096 rows × 12 terms),
plus the existing row-tile scratch buffer of at most 256 KiB. Larger datasets
cache a prefix of complete row tiles and recompute the remainder. Failed cache
allocation falls back to uncached evaluation. This changes neither solver order
nor fitted artifact bytes; the external evaluator API remains unchanged.

`validationFraction` reserves a deterministic seeded holdout. Coefficients and
centering use training rows only; archive selection, polishing and early stopping
use holdout MSE plus `complexityPenalty * complexity`. Without a holdout, selection
uses training MSE. The selected model is not refit on the reserved rows.
Classification preserves the legacy ±1 margin targets, binary/one-vs-rest fitting,
and sigmoid/softmax conversion. These probabilities are not calibrated.
Tree-only controls, non-MSE family losses, hierarchical sources and Polygrad
execution for C-family models raise explicit errors.

New C-family fits write `wlearn.sym.regressor@2` or `wlearn.sym.classifier@2`,
containing a little-endian `SYM2` payload with explicit family kind and semantics
version 2. The payload owns dimensions, protected operators, float32 parameters,
readouts, archive and selection metrics. Unused arithmetic parameters are fixed
to `(p0=1, p1=0)`; unary terms fix their unused second feature to zero. Complexity
counts only active parameters: three arithmetic terms have complexity 7.
Archive identity ignores these unused genes; algebraically equivalent formulas
are not generally deduplicated. AutoML can choose immigrant rates 0, 0.1, 0.2 or 0.3.

Tree artifacts remain `SYM1`/`@1`. Legacy family JSON `@1` remains readable and
keeps its historical metrics and format when re-saved; a successful C refit writes
`@2`. C-family loading and prediction require no Polygrad. Shared Polygrad scoring
remains unimplemented; this is an unreleased checkpoint.

### External family evaluator contract

`src/sym_family.h` exposes an owned search state with propose/accept/finish/free.
A proposal contains descriptors `[candidate, term, 5]` (feature A, feature B,
operator, p0, p1), a row tile and a stable identity. The host returns finite
features `[candidate, row, term]`. C retains train-only moments, centered/scaled
QR, float32 readout rounding and loss reduction in three ordered passes. Neither
Gram-matrix solves nor candidate pre-ranking change the numerical contract.

Candidate batches are bounded to 64 and tiles to 4096 rows. Proposing twice before
accepting returns the same work; invalid submissions leave that work retryable.
Copy borrowed views before async work or Wasm memory growth. Polish submits one
dependent trial at a time. Elites reuse their deterministic fit because data,
holdout mask and objective do not change. Freeing a search cancels it; finishing
transfers the model only after the final proposal reports completion.

This ABI is exercised by independent C, JS and Python feature evaluators. A
Polygrad implementation still needs kernel, transfer-cost and numerical tests;
this interface alone establishes no accelerator performance claim.

## Packages

- npm: `@wlearn/sym`
- PyPI: `wlearn-sym`, imported as `wlearn_sym`
- Bundle typeIds: tree/legacy `wlearn.sym.regressor@1`, `wlearn.sym.classifier@1`, `wlearn.sym.transformer@1`; new family `wlearn.sym.regressor@2`, `wlearn.sym.classifier@2`

## Layout

- `src/`: canonical C11 source; tree search, refinement and persistence are separate files
- `js/csrc/`, `py/csrc/`: generated distribution copies, never versioned
- `js/`: npm package, Emscripten WASM build, browser bundles
- `py/`: Python package with a compiled native extension
- `bench/`: deterministic benchmark harnesses and result JSON
- `references/`: inspected symbolic-regression reference repos
- `test/`: C, JS, browser, Python, and cross-language tests

`../symcpg/` remains the experimental C + Polygrad / JS + Polygrad workspace. Consolidated product code lives in `sym/`; benchmark-only experiments stay outside product claims until they pass broad, honest benchmarks.

## Build And Test

```bash
make -C sym sync-js-csrc sync-py-csrc
make -C sym test-c
cd sym/js && npm run build && npm run build:browser
make -C sym test
make -C sym test-browser
```

For local optional Polygrad parity checks:

```bash
npm --prefix sym/js install
python -m pip install "sym/py[polygrad]"
make -C sym test
```

To test against an unpublished Polygrad checkout, set
`WLEARN_SYM_POLYGRAD_JS=/path/to/polygrad/js` and/or
`WLEARN_POLYGRAD_PY=/path/to/polygrad/py`.

## Benchmarks

```bash
make -C sym bench-friedman
```

Current Friedman-1 comparison uses full-data symbolic fitting by default. A small internal holdout is noisy for structure selection on this deterministic benchmark; use `validationFraction` only when you explicitly want internal validation.

`population=512`, `generations=120`, `maxNodes=63`, `operatorSet=full`, 3 repeats:

| package | rows | mean fit ms | mean R2 | mean bundle bytes |
| --- | ---: | ---: | ---: | ---: |
| `sym` | 512 | 8630.7 | 0.983 | 43277 |
| `xgboost` | 512 | 138.2 | 0.933 | 328323 |
| `sym` | 2048 | 34399.4 | 0.967 | 42285 |
| `xgboost` | 2048 | 148.4 | 0.973 | 340087 |

Interpretation: `sym` can beat XGBoost on 512-row Friedman-1 R2 with compact formulas and is close at 2048 rows, but it is much slower to fit. XGBoost remains the fast predictive baseline.

## Model Surface

- `SymbolicRegressor`: one compact formula optimized for MSE, MAE, or Huber loss.
- `SymbolicClassifier`: binary logistic score or multiclass one-vs-rest score formulas.
- `FormulaTransformer`: top-K frontier formulas as supervised generated features.
- `FormulaVerifier`: honest proof reports: `proved`, `counterexample`, `unknown`, or `unsupported`.

Saved default-engine `.wlrn` bundles contain the raw `SYM1` model blob plus manifest metadata. Bundles load across JS and Python with prediction parity.

JS and Python can also opt into `engine: "pg-family"` for Polygrad summary-kernel family search on regression and classification. Those bundles keep the same `wlearn.sym.*@1` type IDs and set `metadata.engine = "pg-family"`. They are not the default AutoML path.

PG-family supports portable built-in operator subsets through `operatorSet` (`basic`, `smooth`, `full`) or explicit `operators`. Arbitrary JS/Python callables are intentionally not accepted as artifact operators yet.

## Search Controls

The simple path uses one population, full-row scoring, objective-based final
selection, and no archive beyond the frontier.

Optional controls:

- `islands`, `migrationInterval`, `migrationCount`: split search into island
  populations with periodic frontier migration.
- `warmupGenerations`, `warmupMinNodes`: grow allowed formula size over early
  generations.
- `broodSize`: generate multiple children for a parent choice and keep the best
  child.
- `rowSampleSize`: score ordinary offspring on row subsets; frontier candidates
  are rescored on full data before export.
- `localRefineInterval`, `localRefineCount`: run C coefficient refinement on
  promising candidates during search.
- `complexityHofSize`, `complexityBucketWidth`, `finalSelector`: keep a
  complexity-bucket archive and choose the final formula by `objective`, `loss`,
  or `score`.

`frontier()` returns an objective-sorted archive. It is not guaranteed to be a
strict nondominated Pareto front.

Current benchmark notes: island/warmup/brood/subset/local-refine controls improve
Friedman-1/2/3 quality versus the copied `symc` baseline. Complexity HOF is
available but currently neutral under `finalSelector: 'objective'`; `loss` and
`score` are explicit tuning options, not defaults.

## Local Polygrad 0.6 migration

This checkout targets the local Polygrad 0.6.0 candidate. Publication and the
final fixed dependency pin await upstream release acceptance. Use the local
checkout overrides above while developing; the install commands apply after
publication.

Formula execution and refinement use the public Model capture API with float32
inputs, targets and parameters. Protected operations replace non-finite node
results with zero and clamp each node to ±1e12, matching the C evaluation rules
within float32 precision. Execution can therefore differ from C double precision
near discontinuities or for large trigonometric arguments.

Refinement reads updated constants from Model-owned state and evaluates a copy
with the C backend before committing. Failed or worse candidates leave the
original model unchanged. Only the selected regression formula (`index=0`) is
refinable. Original search-time metrics remain attached to the formula; the
returned report measures MSE on the supplied refinement data and does not claim
new held-out validation coverage. An installed backend failure raises an error;
`unsupported` is reserved for unsupported operators or model strategies.

Caller-supplied runtimes are borrowed. Temporary Models are disposed after
execution/refinement; internally created runtimes are disposed by their owner.


### C search stepping

`src/sym.h` exposes `sym_search_new`, `sym_search_propose`,
`sym_search_score`, `sym_search_accept`, `sym_search_finish` and
`sym_search_free`. Ordinary `sym_fit` drives this same state machine internally.
The search owns copies of training inputs and yields bounded scoring batches;
acceptance validates the complete batch before updating candidates. Batch IDs
reject stale results. `finish` transfers the fitted model exactly once.

The current implementation supports tree search with the C scorer. Local
coefficient refinement and final selection remain in C. Separating family
search and adding a Polygrad scorer to this API are planned work; the existing
`engine: 'pg-family'` path still uses its frontend implementation.

Batch formula/data views are borrowed until acceptance or destruction. A host
performing asynchronous evaluation must copy views before yielding and protect
the search lifetime while work is pending. `wl_sym_search_*` exposes packed
frontend buffers without depending on C structure padding. These low-level
interfaces are under development before the initial release.

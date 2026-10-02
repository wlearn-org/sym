# sym Benchmarks

Benchmarks are separate from unit tests. They use deterministic datasets and write JSON result files for change-to-change comparison.

## Quality screening and in-loop polish

`quality.py` runs the versioned 24-task `quality-tasks.json` manifest: synthetic
equations with noise, irrelevant columns, input/target rescaling, collinearity,
extrapolation, and scikit-learn's diabetes dataset. This is a screening set, not
a substitute for SRBench or evidence of general superiority. Generated target
formulas never initialize or constrain model search.

From this repository, with NumPy, scikit-learn and wlearn installed:

```sh
make build-c
PYTHONPATH=py python -m pytest bench/test_quality.py -q
SYM_LIB_PATH="$PWD/build/libsym.so" PYTHONPATH=py \
  python bench/quality.py --jobs 2 --output bench/results/quality.jsonl
```

The default arms are family search without polish, final polish, in-loop plus
final polish, tree search, standardized ridge and histogram gradient boosting.
The `during` control reuses coordinate proposals with QR readout refits.
Additional arms `scaled`, `lm`, `scaled-lm` isolate train-only parameter scaling
and finite-difference profiled damped least squares, using the same in-loop
schedule. `lm` is a bounded damped step, not MINPACK or an analytic-gradient
implementation. Keep final-fit and total pilot costs distinct when comparing it.
Five search seeds see identical training/test data per task. Test data never
enter fitting or timing pilots. Each Sym result is saved/reloaded and checked
against an independent NumPy evaluator of its exported formula.

`--budget-ms 100 --arms family,final,during` calibrates generations using up to
three training-only timing pilots per arm. It reports pilot cost separately and
flags final fits outside ±25% of the target as unmatched. This compares final-fit
cost, not equal total compute including calibration. Fixed-generation runs also
report actual cost; extra refinement is not free. Preset sklearn/PySR/Operon
runs are never labeled budget-matched. Keep failures and unmatched results in
reports, and restrict paired budget claims to pairs meeting the stated limit.

Optional arms: `family-pg`, `during-pg`, `post-adam`, `pysr`, `operon`. These require
their respective packages. Polygrad/PySR runs require `--jobs 1`; each worker
limits numerical-library threads to one. Cap total CPU affinity to half the
machine when running additional checks alongside this harness. `--device` is
passed to Polygrad's public runtime; there is no additional device environment
variable. Timing distinguishes a first fit in a fresh process from a fit after
timing pilots. Runtime creation, prediction, pilots and fit times are separate.

Worker timeouts cover imports and qualification as well as fitting, terminate
the complete process group and remain recorded as failures. RSS is the isolated
worker's process watermark, including imports and pilots, not GPU memory or a
model-only allocation estimate. Error metrics are numerical; no exact symbolic
recovery claim is inferred from R². JSONL rows and the adjacent metadata file
record parameters, versions, task manifest identity and failures.

## Friedman JS

```bash
make -C sym bench-friedman
```

Direct invocation:

```bash
node sym/bench/friedman-js.js \
  --datasets 1,2,3 \
  --rows 128,512,2048 \
  --predictRows 1024 \
  --population 128 \
  --generations 30 \
  --maxNodes 31 \
  --operatorSet full \
  --include sym,symc \
  --repeat 3 \
  --output sym/bench/results/friedman-js.latest.json
```

Use `--include sym,xgboost` to compare against local `@wlearn/xgboost`. XGBoost defaults in this harness are `numRound=200`, `max_depth=4`, and `eta=0.05`; override with `--xgbNumRound`, `--xgbMaxDepth`, and `--xgbEta`.

`validationFraction` defaults to `0` in `sym`. Use `--validationFraction <fraction>` only when you intentionally want an internal holdout; small holdouts are noisy for symbolic structure selection on deterministic Friedman-1.

Current full-data comparison artifact:
`bench/results/friedman-js.sym-coeffopt-xgboost-20260628.json`.

Current Friedman-2/3 comparison artifact:
`bench/results/friedman-js.2-3-sym-coeffopt-xgboost-20260628.json`.

Optional Polygrad constant adaptation:

```bash
node sym/bench/friedman-js.js \
  --rows 512 \
  --population 512 \
  --generations 120 \
  --maxNodes 63 \
  --operatorSet full \
  --refinePolygrad true \
  --refineEpochs 80 \
  --refineLr 0.001
```

Refinement reports include `refineStatus`, `refineReason`, and
`refineCommitted`, plus before/after training loss when refinement runs.
`unsupported` means no constants were committed; `rejected` means refinement ran
but made training MSE worse or non-finite; `improved` means constants were
committed because training MSE improved.

Datasets:

```text
Friedman-1:
  y = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 + noise

Friedman-2:
  y = sqrt(x0^2 + (x1 x2 - 1 / (x1 x3))^2) + noise

Friedman-3:
  y = atan((x1 x2 - 1 / (x1 x3)) / x0) + noise
```

Reported fields:

- `fitMs`
- `refineMs`
- `refineStatus`
- `refineReason`
- `refineCommitted`
- `refineBeforeLoss`
- `refineAfterLoss`
- `refineHistoryFirst`
- `refineHistoryLast`
- `predictMs`
- `r2`
- `formulaNodes`
- `formulaText`
- `bundleBytes`

`sym` is the current C-first product package. `symc` is the older copied C baseline retained for comparison.

## Shared family backends (2026-09-28)

`family-backends.py` compares complete C and public-Polygrad fits with matched
seeds, independent test rows, batched polish, and optional hierarchy/refinement.
The committed `results/fit-*-20260928.json` files contain raw measurements.

On an RTX 3060 Laptop, Python 3.11.12, native single-thread C versus CUDA float64,
4096 rows / population 256 / 120 generations / 8 terms took about 30–34 seconds
in C and 15–16 seconds with Polygrad. All six test prediction comparisons matched
exactly. Each fit creates its runtime and includes graph setup; repeated runs
share a process and can reuse compiler/driver caches. This is not a fresh-process
cold-start comparison or a reproduction of the historical symcpg 10× result.

At 512 rows / population 64 / 10 generations / 4 terms, C took about 36–39 ms
following initial library loading; Polygrad took about 2.8 seconds. C remains
the default. The hierarchy/refinement ablation did not consistently improve test
quality; only one of six distinct refinement trials was accepted. Both features
remain optional. These two synthetic datasets do not establish general quality.

Run with local Python packages and a freshly built Sym library:

```sh
python bench/family-backends.py --rows 4096 --population 256 --generations 120 --terms 8 --output results.json
sh bench/scorer/build.sh
POLY_DEV=CUDA python -m pytest bench/scorer/test_scorer.py
```

`scorer/` contains diagnostic C-reference adapters and fixed-candidate tests;
its Python numerical scorer delegates to the package implementation. Its isolated
kernel timings are not complete-fit speedups. The recorded fit measurements
precede final boundary-validation and environment-selection fixes; their explicit
CUDA device and numerical algorithm are unchanged by those fixes.

## Fresh and reused runtimes (2026-09-28 follow-up)

Historical measurements and failure reports below predate published Polygrad
0.6.0. They are not a current failure report; rerun the backend tests against the
installed version when evaluating numerical correctness and ownership.

`family-backends.py` now accepts `--batches`, `--dtypes`, `--datasets` and
`--warm-repeats`. Each configuration reports runtime creation separately, a
complete first fit, then complete subsequent fits sharing that runtime. Graph
construction remains included. These are not fresh-process cold measurements.

RTX3060Laptop, native single-thread C, Friedman1 seed11,4096rows,population256,
8terms,2polish passes, independent512test rows:

| Configuration | 30 generations | 120 generations |
|---|---:|---:|
| C |7.21s|28.57s|
| f64,batch64,fresh |8.38s|15.71s|
| f64,batch64,warm |2.54s|10.27s|
| f32,batch256,fresh |6.62s|9.12s|
| f32,batch256,warm |1.28s|4.78s|

All warm fits had0runtime-cache misses. Runtime allocation itself was milliseconds;
most of the extra first-fit cost was graph setup/compilation. f64/batch256 was
slightly slower warm than batch64 (2.88/10.87s). The population-based default
reduces padding on small populations and crossings on large ones; larger batches
use more memory and are not a universal speed improvement.

f64 matched C predictions in these runs. f32/batch256 maximum prediction delta
was5.82e-5 at30generations but0.471 at120, with almost identical testR². Numeric
perturbations change selection trajectories. Four fixed-candidate f32 numerical
cases passed1e-5 tolerances; the large-offset/cosine case correctly rejects f32.
Native f64 remains the default. Extended f64 offset testing also exposed a CPU
sine-lowering error (max prediction delta0.000382); CUDA passed. Its regression
remains failing on CPU, rather than relaxing the tolerance.

`family-browser.js` reports real browser WASM C versus public WebGPU on a matched
LCG Friedman1 dataset (different generator from the native script). At4096rows,
256candidates,30generations,8terms,batch256: C8.30s,WebGPU fresh5.02s,warm2.97s;
max prediction delta1.32e-4. At512rows,64candidates,10generations,4terms,batch64: C62ms,WebGPU fresh2.61s,
warm152ms. These are browser measurements, not inferred from CUDA. Run with the
display/Chromium setup used by browser tests:

```sh
POLY_DEV=webgpu node bench/family-browser.js --rows 4096 --population 256 --generations 30 --terms 8 --batch 256 --output browser.json
```

Raw follow-up measurements: `results/reuse-*-20260928.json`.

An implicit process-global cache is not added. Repeated disposed Sym fits retain
runtime buffers: a256row/16candidate/3term Python CUDA probe retained527488bytes
per fit even after GC,schedule-cache clearing and collection; JS also grows.
Standalone add,QR,custom-kernel,chained add/QR and copy/replay probes reclaim all
buffers, so the precise owner is unresolved. A caller-owned runtime bounds its
lifetime and supports reuse across AutoML folds via the tested class factory.
Long-lived reuse needs the ownership issue resolved, and scorer-level reuse must
replace data-dependent constants with dynamic inputs or include them in its key.

### Target-unit experiment (October 2)

`quality-scale-tasks.json` provides eight fresh noisy functions at three target
scales, with identical observations across rescalings. Select it with
`--manifest bench/quality-scale-tasks.json`; compare `scaled-lm` and
`scaled-lm-relative`. The latter enables family `lossScale: 'target-variance'`.
Neither supplies target formulas to the estimator.

All 720 fits passed across the existing screen, fresh duration-target confirmation
and fixed-generation confirmation. On small-target confirmation cases, relative
loss won 35, lost 1 and tied 4 at matched final-fit duration (0.001 NMSE threshold).
At ordinary scale: 8/10/22. Timing pilots cost extra. Keep the policy opt-in;
these limited synthetic comparisons do not establish a universal default.
Commands, full results and numerical caveats:
`../temp/runs/20261002-sym-loss-scale/README.md` (local workspace evidence).

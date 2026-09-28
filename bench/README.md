# sym Benchmarks

Benchmarks are separate from unit tests. They use deterministic datasets and write JSON result files for change-to-change comparison.

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

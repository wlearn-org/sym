"""Complete-estimator timings; setup and compilation are included in every fit."""

import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
from wlearn_sym import SymbolicRegressor


def dataset(name, rows, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (rows, 5)).astype(np.float32).astype(float)
    if name == "friedman1":
        y = (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
        )
    else:
        X = X * 2 - 1
        y = np.sin(2.3 * X[:, 0] + 0.4) + X[:, 1] * X[:, 2] + 0.2 * X[:, 3]
    return X, y


def timed_fit(X, y, Xt, yt, params, refine):
    model = SymbolicRegressor(params)
    try:
        start = time.perf_counter()
        model.fit(X, y)
        fit_ms = (time.perf_counter() - start) * 1000
        pred = model.predict(Xt)
        row = dict(
            fit_ms=fit_ms,
            test_r2=float(1 - np.sum((pred - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)),
            bytes=len(model.save()),
        )
        if refine:
            start = time.perf_counter()
            report = model.refine_polygrad(X, y, epochs=refine)
            row.update(
                refine_ms=(time.perf_counter() - start) * 1000,
                refined_test_r2=float(model.score(Xt, yt)),
                committed=report["committed"],
            )
        return row, pred
    finally:
        model.dispose()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--terms", type=int, default=4)
    p.add_argument("--seeds", default="11,23,37")
    p.add_argument("--datasets", default="smooth,friedman1")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--batches", default="auto", help="Comma-separated capacities, or auto"
    )
    p.add_argument("--dtypes", default="float64", help="float64,float32 or either")
    p.add_argument("--warm-repeats", type=int, default=1)
    p.add_argument("--hierarchical", action="store_true")
    p.add_argument("--polish", type=int, default=2)
    p.add_argument("--refine", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.warm_repeats < 1:
        p.error(
            "--warm-repeats must be positive; report both fresh and reused runtime fits"
        )
    batches = [None if v == "auto" else int(v) for v in a.batches.split(",")]
    dtypes = a.dtypes.split(",")
    if any(v not in ("float32", "float64") for v in dtypes):
        p.error("--dtypes must contain float32 or float64")
    names = a.datasets.split(",")
    if any(v not in ("smooth", "friedman1") for v in names):
        p.error("--datasets must contain smooth or friedman1")
    import polygrad
    import os

    result = dict(
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            polygrad=getattr(polygrad, "__version__", None),
            poly_dev=os.environ.get("POLY_DEV"),
            dev=os.environ.get("DEV"),
        ),
        settings=vars(a),
        timing=(
            "fresh: runtime creation plus complete first fit; warm: complete subsequent fit "
            "on the same runtime/data/seed, including scorer reconstruction. Compilation "
            "caches may survive in the process/driver even for fresh runtimes; not process-cold. "
            "Runtime disposal and optional gradient refinement excluded from total_ms."
        ),
        results=[],
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        result["results"].append(row)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    for name in names:
        for seed in map(int, a.seeds.split(",")):
            X, y = dataset(name, a.rows, seed + 1000)
            Xt, yt = dataset(name, 512, seed + 2000)
            params = dict(
                strategy="family",
                population=a.population,
                generations=a.generations,
                terms=a.terms,
                seed=seed,
                eliteCount=8,
                validationFraction=0.2,
                polishPasses=a.polish,
                polishBatchSize=32 if a.polish else 0,
                hierarchical=a.hierarchical,
            )
            c, reference = timed_fit(X, y, Xt, yt, {**params, "backend": "c"}, a.refine)
            emit(dict(dataset=name, seed=seed, backend="c", total_ms=c["fit_ms"], **c))
            for dtype in dtypes:
                for batch in batches:
                    begin = time.perf_counter()
                    runtime = polygrad.create(device=a.device)
                    runtime_ms = (time.perf_counter() - begin) * 1000
                    options = dict(
                        params, backend="polygrad", polygrad=runtime, scorerDtype=dtype
                    )
                    if batch is not None:
                        options["batchSize"] = batch
                    try:
                        for repeat in range(a.warm_repeats + 1):
                            before = runtime.stats()
                            row, pred = timed_fit(X, y, Xt, yt, options, a.refine)
                            after = runtime.stats()
                            counters = {
                                k: after[k] - before[k]
                                for k in after
                                if isinstance(after[k], (int, float)) and k in before
                            }
                            emit(
                                dict(
                                    dataset=name,
                                    seed=seed,
                                    backend="polygrad",
                                    dtype=dtype,
                                    batch=batch or min(a.population, 512),
                                    batch_setting="auto" if batch is None else batch,
                                    phase="fresh" if repeat == 0 else "warm",
                                    repeat=repeat,
                                    runtime_create_ms=runtime_ms if repeat == 0 else 0,
                                    total_ms=row["fit_ms"]
                                    + (runtime_ms if repeat == 0 else 0),
                                    runtime_counters=counters,
                                    max_prediction_delta_from_c=float(
                                        np.max(np.abs(pred - reference))
                                    ),
                                    **row,
                                )
                            )
                    finally:
                        runtime.dispose()


if __name__ == "__main__":
    main()

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--terms", type=int, default=4)
    p.add_argument("--seeds", default="11,23,37")
    p.add_argument("--device", default="CUDA")
    p.add_argument("--hierarchical", action="store_true")
    p.add_argument("--polish", type=int, default=2)
    p.add_argument("--refine", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = dict(
        environment=dict(
            python=platform.python_version(), platform=platform.platform()
        ),
        settings=vars(a),
        timing="Complete fit, fresh estimator/runtime; includes device setup and compilation; no prewarm.",
        results=[],
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in ("smooth", "friedman1"):
        for seed in map(int, a.seeds.split(",")):
            X, y = dataset(name, a.rows, seed + 1000)
            Xt, yt = dataset(name, 512, seed + 2000)
            predictions = {}
            for backend in ("c", "polygrad"):
                params = dict(
                    strategy="family",
                    backend=backend,
                    population=a.population,
                    generations=a.generations,
                    terms=a.terms,
                    seed=seed,
                    eliteCount=8,
                    validationFraction=0.2,
                    polishPasses=a.polish,
                    polishBatchSize=32 if a.polish else 0,
                    hierarchical=a.hierarchical,
                    batchSize=64,
                    polygrad=dict(device=a.device),
                )
                model = SymbolicRegressor(params)
                try:
                    start = time.perf_counter()
                    model.fit(X, y)
                    fit_ms = (time.perf_counter() - start) * 1000
                    pred = model.predict(Xt)
                    predictions[backend] = pred
                    row = dict(
                        dataset=name,
                        seed=seed,
                        backend=backend,
                        fit_ms=fit_ms,
                        test_r2=float(
                            1 - np.sum((pred - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)
                        ),
                        bytes=len(model.save()),
                    )
                    if a.refine:
                        start = time.perf_counter()
                        report = model.refine_polygrad(
                            X, y, epochs=a.refine, polygrad=dict(device=a.device)
                        )
                        row.update(
                            refine_ms=(time.perf_counter() - start) * 1000,
                            refined_test_r2=float(model.score(Xt, yt)),
                            committed=report["committed"],
                        )
                    if backend == "polygrad":
                        row["max_prediction_delta_from_c"] = float(
                            np.max(np.abs(pred - predictions["c"]))
                        )
                    result["results"].append(row)
                    out.write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                finally:
                    model.dispose()


if __name__ == "__main__":
    main()

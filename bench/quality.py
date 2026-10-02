"""Isolated complete-fit quality screening; optional train-only timing calibration.

The worker timeout covers imports, timing pilots, fitting and artifact verification.
Fit budgets are targets, not deadlines: unmatched cases are explicitly reported.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from quality_data import MANIFEST, dataset, manifest


ARMS = (
    "family",
    "final",
    "during",
    "scaled",
    "lm",
    "scaled-lm",
    "tree",
    "ridge",
    "histgb",
    "family-pg",
    "during-pg",
    "post-adam",
    "pysr",
    "operon",
)


def fit_model(arm, X, y, seed, generations, settings, runtime=None):
    from wlearn_sym import SymbolicRegressor

    params = dict(
        population=settings["population"],
        eliteCount=4,
        generations=generations,
        seed=seed,
        validationFraction=0.2,
        strategy="tree" if arm == "tree" else "family",
    )
    if arm != "tree":
        params["terms"] = settings["terms"]
    if arm in ("final", "during", "during-pg", "scaled"):
        params.update(polishPasses=2, polishBatchSize=16)
    if arm in ("during", "during-pg", "scaled", "lm", "scaled-lm"):
        params.update(localRefineInterval=3, localRefineCount=2)
    if arm in ("scaled", "scaled-lm"):
        params.update(scaleAware=True)
    if arm in ("lm", "scaled-lm"):
        params.update(polishMethod="lm", polishPasses=2)
    if arm.endswith("-pg"):
        params.update(backend="polygrad", polygrad=runtime)
    model = SymbolicRegressor(params)
    try:
        begin = time.perf_counter()
        model.fit(X, y)
        refinement = None
        if arm == "post-adam":
            refinement = model.refine_polygrad(X, y, epochs=20, polygrad=runtime)
        elapsed = 1000 * (time.perf_counter() - begin)
        return (
            model,
            elapsed,
            {k: v for k, v in params.items() if k != "polygrad"},
            refinement,
        )
    except BaseException:
        model.dispose()
        raise


def worker(case):
    import numpy as np
    import resource

    task, arm, seed, settings = (case[k] for k in ("task", "arm", "seed", "settings"))
    X, y, Xt, yt = dataset(task, settings["rows"], settings["test_rows"])
    model = runtime = loaded = None
    result = dict(case, status="ok", rows=len(y), test_rows=len(yt))
    try:
        create_ms = 0
        if arm.endswith("-pg") or arm == "post-adam":
            import polygrad

            start = time.perf_counter()
            runtime = polygrad.create(device=settings["device"])
            create_ms = (time.perf_counter() - start) * 1000
        generations = settings["generations"]
        pilots = []
        if arm in ("ridge", "histgb"):
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            model = (
                make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                if arm == "ridge"
                else HistGradientBoostingRegressor(max_iter=100, random_state=seed)
            )
            start = time.perf_counter()
            model.fit(X, y)
            fit_ms = 1000 * (time.perf_counter() - start)
            result["params"] = {
                "preset": "standardized ridge alpha=1"
                if arm == "ridge"
                else "histgb max_iter=100"
            }
        elif arm in ("pysr", "operon"):
            # External engines are baselines only, never runtime dependencies.
            if arm == "pysr":
                from pysr import PySRRegressor

                model = PySRRegressor(
                    niterations=generations,
                    populations=1,
                    population_size=settings["population"],
                    binary_operators=["+", "-", "*", "/"],
                    unary_operators=["sin", "cos", "tanh"],
                    parallelism="serial",
                    deterministic=True,
                    random_state=seed,
                    progress=False,
                    verbosity=0,
                    temp_equation_file=True,
                    delete_tempfiles=True,
                )
            else:
                from pyoperon.sklearn import SymbolicRegressor as Operon

                model = Operon(
                    generations=generations,
                    population_size=settings["population"],
                    random_state=seed,
                    n_threads=1,
                )
            start = time.perf_counter()
            model.fit(X, y)
            fit_ms = 1000 * (time.perf_counter() - start)
            result["params"] = {
                "generations": generations,
                "population": settings["population"],
            }
        else:
            # Pilots choose duration only, never inspect held-out quality. Always
            # report their cost and separate calibrated/warm from fresh fits.
            if settings["budget_ms"]:
                for _ in range(3):
                    pilot, ms, _, _ = fit_model(
                        arm, X, y, seed, generations, settings, runtime
                    )
                    pilot.dispose()
                    pilots.append(dict(generations=generations, fit_ms=ms))
                    if abs(ms / settings["budget_ms"] - 1) <= 0.2:
                        break
                    proposed = min(
                        512,
                        max(
                            1,
                            round(generations * settings["budget_ms"] / max(ms, 0.01)),
                        ),
                    )
                    if proposed == generations:
                        break
                    generations = proposed
            model, fit_ms, result["params"], refinement = fit_model(
                arm, X, y, seed, generations, settings, runtime
            )
            result["refinement_committed"] = refinement and refinement["committed"]
        result.update(
            fit_ms=fit_ms,
            runtime_create_ms=create_ms,
            timing_pilots=pilots,
            calibration_ms=sum(p["fit_ms"] for p in pilots),
            timing_phase="after timing pilots"
            if pilots
            else "first fit in fresh process",
        )
        # Reference models have different stopping controls; do not label their
        # preset runtimes as matched just because a Sym budget was requested.
        result["budget_matched"] = (
            abs(fit_ms / settings["budget_ms"] - 1) <= 0.25
            if settings["budget_ms"]
            and arm not in ("ridge", "histgb", "pysr", "operon")
            else None
        )
        begin = time.perf_counter()
        pred = np.asarray(model.predict(Xt)).reshape(-1)
        result["predict_ms"] = (time.perf_counter() - begin) * 1000
        if pred.shape != yt.shape or not np.isfinite(pred).all():
            raise ValueError("nonfinite or incorrectly shaped test prediction")
        mse = float(np.mean((pred - yt) ** 2))
        result.update(
            test_mse=mse,
            test_nmse=mse / float(np.var(yt)),
            test_r2=1 - mse / float(np.var(yt)),
        )
        if hasattr(model, "save"):
            from wlearn_sym import SymbolicRegressor

            blob = model.save()
            loaded = SymbolicRegressor.load(blob)
            np.testing.assert_array_equal(pred, loaded.predict(Xt))
            formula = model.formula()
            from quality_reference import predict as reference_predict

            reference = reference_predict(formula, Xt)
            np.testing.assert_allclose(pred, reference, atol=1e-5, rtol=1e-5)
            result.update(
                bundle_bytes=len(blob),
                bundle_sha256=hashlib.sha256(blob).hexdigest(),
                formula=formula,
                formula_text=model.formula(format="text"),
                complexity=formula.get("complexity"),
                artifact_roundtrip=True,
                reference_max_delta=float(np.max(abs(pred - reference))),
            )
        # Linux reports KiB. This is the whole isolated worker watermark, not an
        # allocation delta attributed solely to the estimator or GPU residency.
        result["process_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        if runtime:
            result["runtime_stats_before_dispose"] = runtime.stats()
        return result
    finally:
        for obj in (loaded, model, runtime):
            if obj is not None and hasattr(obj, "dispose"):
                obj.dispose()


def execute(case, timeout):
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        JULIA_NUM_THREADS="1",
    )
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(json.dumps(case), timeout=timeout)
        except subprocess.TimeoutExpired:
            # Julia and native helpers may spawn children; terminate the whole
            # worker group so a timeout cannot leave benchmark work running.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return dict(case, status="timeout", timeout_s=timeout)
        marker = [
            line[7:] for line in stdout.splitlines() if line.startswith("RESULT ")
        ]
        if proc.returncode != 0 or not marker:
            return dict(
                case,
                status="error",
                error=(stderr or stdout)[-4000:],
                returncode=proc.returncode,
            )
        return json.loads(marker[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--arms", default="family,final,during,tree,ridge,histgb")
    p.add_argument("--tasks", default="all")
    p.add_argument("--seeds", default="11,23,37,51,73")
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--test-rows", type=int, default=512)
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--terms", type=int, default=4)
    p.add_argument("--budget-ms", type=float, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--output")
    a = p.parse_args()
    if a.worker:
        case = json.loads(sys.stdin.read())
        try:
            result = worker(case)
        except Exception as error:
            result = dict(
                case, status="error", error=f"{type(error).__name__}: {error}"
            )
        print("RESULT " + json.dumps(result, allow_nan=False), flush=True)
        return
    if not a.output:
        p.error("--output is required")
    arms = a.arms.split(",")
    if any(arm not in ARMS for arm in arms):
        p.error(f"--arms must be selected from {ARMS}")
    if a.jobs < 1 or a.jobs > max(1, (os.cpu_count() or 1) // 2):
        p.error("--jobs must use at most half the host CPUs")
    if a.jobs != 1 and any(
        arm.endswith("-pg") or arm in ("post-adam", "pysr") for arm in arms
    ):
        p.error("Polygrad and PySR measurements require --jobs 1")
    if (
        min(a.rows, a.test_rows, a.generations, a.terms) < 1
        or a.population < 8
        or a.budget_ms < 0
        or a.timeout <= 0
    ):
        p.error("invalid size, budget or timeout")
    tasks = manifest()["tasks"]
    if a.tasks != "all":
        wanted = a.tasks.split(",")
        if set(wanted) - {t["id"] for t in tasks}:
            p.error("unknown task ID")
        tasks = [t for t in tasks if t["id"] in wanted]
    settings = {
        k: getattr(a, k)
        for k in (
            "rows",
            "test_rows",
            "population",
            "generations",
            "terms",
            "budget_ms",
            "device",
        )
    }
    cases = [
        dict(task=t, arm=arm, seed=seed, settings=settings)
        for t in tasks
        for seed in map(int, a.seeds.split(","))
        for arm in arms
    ]
    versions = {}
    for name in ("numpy", "scikit-learn", "polygrad", "pysr", "pyoperon"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        "bench/quality.py",
        "bench/quality_data.py",
        "bench/quality_reference.py",
        "src/sym_family.c",
        "src/sym_family_lm.c",
        "src/sym_family.h",
        "src/wl_api.c",
        "py/wlearn_sym/_family_c.py",
    ]
    source_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in source_paths
    }
    native_library = os.environ.get("SYM_LIB_PATH")
    if native_library:
        source_hashes[native_library] = hashlib.sha256(
            Path(native_library).read_bytes()
        ).hexdigest()
    metadata = dict(
        settings=vars(a),
        python=sys.version,
        platform=platform.platform(),
        affinity=sorted(os.sched_getaffinity(0)),
        versions=versions,
        source_sha256=source_hashes,
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        tasks=tasks,
        cases=len(cases),
        limitations="Numerical test error, not proof of equation recovery. RSS includes imports/pilots. Budgets target fit time; pilots and unmatched cases are reported. External baselines use fixed presets.",
    )
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    with output.open("w") as stream, ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for i, row in enumerate(pool.map(lambda c: execute(c, a.timeout), cases), 1):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                f"{i}/{len(cases)} {row['task']['id']} {row['arm']} seed={row['seed']} {row['status']} "
                f"nmse={row.get('test_nmse')} fit_ms={row.get('fit_ms')}",
                flush=True,
            )


if __name__ == "__main__":
    main()

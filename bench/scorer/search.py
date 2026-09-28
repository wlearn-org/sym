"""Full C-owned search with a benchmark-only reduced-result bridge."""

import argparse
import ctypes as ct
import json
import time

import numpy as np
from resident import Reference, ResidentScorer, _ptr
from run import data


class Search:
    def __init__(self, X, y, population, generations, terms, seed, polish):
        self.ref = Reference()
        self.lib = self.ref.lib
        self.X, self.y = X, y
        self.terms = terms
        self.args = (
            _ptr(X),
            _ptr(y),
            len(X),
            X.shape[1],
            population,
            generations,
            terms,
            seed,
            polish,
        )
        args = [ct.c_void_p] * 2 + [ct.c_int] * 5 + [ct.c_uint32, ct.c_int]
        for name in ("scorer_search_new", "scorer_fit"):
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, ct.c_void_p
        self.lib.scorer_search_mask.argtypes = [ct.c_void_p] * 2
        self.lib.scorer_search_next.argtypes = [ct.c_void_p] * 3
        self.lib.scorer_search_accept.argtypes = [ct.c_void_p, ct.c_uint32] + [
            ct.c_void_p
        ] * 2
        self.lib.sym_family_search_finish.argtypes = [ct.c_void_p]
        self.lib.sym_family_search_finish.restype = ct.c_void_p
        self.lib.sym_family_search_free.argtypes = [ct.c_void_p]
        self.lib.sym_family_free.argtypes = [ct.c_void_p]
        self.lib.sym_family_predict.argtypes = (
            [ct.c_void_p] * 2 + [ct.c_int] * 3 + [ct.c_void_p]
        )
        self.handle = self.lib.scorer_search_new(*self.args)
        if not self.handle:
            raise RuntimeError("C search creation failed")
        self.mask = np.empty(len(X), np.uint8)
        self.lib.scorer_search_mask(self.handle, _ptr(self.mask))
        self.descriptors = np.zeros((64, terms, 5))
        self.id = ct.c_uint32()

    def next(self):
        count = self.lib.scorer_search_next(
            self.handle, _ptr(self.descriptors), ct.byref(self.id)
        )
        if count < 0:
            raise RuntimeError("C propose failed")
        if count:
            # Fixed device shape; padded candidates are never accepted by search.
            self.descriptors[count:] = self.descriptors[0]
        return count

    def accept(self, coefficients, losses):
        coefficients = np.ascontiguousarray(coefficients, dtype=float)
        losses = np.ascontiguousarray(losses, dtype=float)
        rc = self.lib.scorer_search_accept(
            self.handle, self.id, _ptr(coefficients), _ptr(losses)
        )
        if rc:
            raise RuntimeError(f"C reduced accept failed: {rc}")

    def predict_and_free(self, model, X):
        if not model:
            raise RuntimeError("C fit/finish failed")
        try:
            prediction = np.empty(len(X))
            rc = self.lib.sym_family_predict(
                model, _ptr(X), len(X), X.shape[1], 0, _ptr(prediction)
            )
            if rc:
                raise RuntimeError("C prediction failed")
            return prediction
        finally:
            self.lib.sym_family_free(model)

    def dispose(self):
        if self.handle:
            self.lib.sym_family_search_free(self.handle)
            self.handle = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--population", type=int, default=128)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--terms", type=int, default=6)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--polish", type=int, default=0)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument(
        "--small-batch-c",
        type=int,
        default=0,
        help="explicit hybrid: score up to this many candidates in C",
    )
    p.add_argument("--device", default="CUDA")
    p.add_argument(
        "--mode", choices=["custom", "tensor", "reference"], default="custom"
    )
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    a = p.parse_args()
    X, y, _, _ = data(a.rows, 64, a.terms, "ordinary", a.seed)
    Xt, yt, _, _ = data(1024, 64, a.terms, "ordinary", a.seed + 100)
    setup_search = Search(X, y, a.population, a.generations, a.terms, a.seed, a.polish)
    scorer = None
    try:
        setup_search.next()
        if a.mode != "reference":
            scorer = ResidentScorer(
                X,
                y,
                setup_search.mask,
                setup_search.descriptors,
                device=a.device,
                mode=a.mode,
                dtype=a.dtype,
            )
            # Exercise first readback/host solve/loss too, outside warm-fit timings.
            warmup_start = time.perf_counter()
            scorer.score(setup_search.descriptors)
            print(
                json.dumps(
                    {
                        "event": "setup",
                        "ms": scorer.setup_ms,
                        "first_score_ms": (time.perf_counter() - warmup_start) * 1000,
                        "settings": vars(a),
                    }
                ),
                flush=True,
            )
        for repeat in range(a.repeat):
            begin = time.perf_counter()
            model = setup_search.lib.scorer_fit(*setup_search.args)
            c_ms = (time.perf_counter() - begin) * 1000
            pred_c = setup_search.predict_and_free(model, Xt)
            begin = time.perf_counter()
            search = Search(
                X, y, a.population, a.generations, a.terms, a.seed, a.polish
            )
            calls = evaluated = c_calls = 0
            try:
                while count := search.next():
                    if scorer and count > a.small_batch_c:
                        result = scorer.score(search.descriptors)
                        coef, losses = result["coefficients"], result["losses"]
                    else:
                        coef, losses, _ = search.ref.fit(
                            X,
                            y,
                            search.mask,
                            np.ascontiguousarray(search.descriptors[:count]),
                            1e-8,
                        )
                        c_calls += 1
                    search.accept(coef, losses)
                    calls += 1
                    evaluated += count
                model = search.lib.sym_family_search_finish(search.handle)
                fit_ms = (time.perf_counter() - begin) * 1000
                pred = search.predict_and_free(model, Xt)
            finally:
                search.dispose()
            print(
                json.dumps(
                    {
                        "event": "fit",
                        "repeat": repeat,
                        "c_ms": c_ms,
                        "fit_ms": fit_ms,
                        "calls": calls,
                        "c_calls": c_calls,
                        "evaluated": evaluated,
                        "prediction_rmse_vs_c": float(
                            np.sqrt(np.mean((pred - pred_c) ** 2))
                        ),
                        "test_mse_c": float(np.mean((pred_c - yt) ** 2)),
                        "test_mse": float(np.mean((pred - yt) ** 2)),
                    }
                ),
                flush=True,
            )
    finally:
        setup_search.dispose()
        if scorer:
            scorer.dispose()


if __name__ == "__main__":
    main()

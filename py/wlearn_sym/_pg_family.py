"""Optional Polygrad-backed formula-family search for wlearn_sym.

The default wlearn_sym product path is the C core. This module mirrors the JS
`engine: "pg-family"` artifact and API surface so JS/Python bundles can round
trip. Fitting imports Polygrad lazily; loaded models predict from saved formula
state without requiring Polygrad.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from ._family_format import (OPS, PG_FAMILY_MEDIA, resolve_operator_ids, Candidate,
                             _candidate_to_plain, _candidate_from_plain, _formula_text)


def _lcg(seed: int):
    state = int(seed) & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / 0x100000000

    return rnd


def _rand_int(rng, n: int) -> int:
    return int(math.floor(rng() * n))


def _rand_range(rng, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * rng())


def normalize_x(X, n_features: Optional[int] = None):
    if isinstance(X, dict) and {"rows", "cols", "data"} <= set(X):
        rows = int(X["rows"])
        cols = int(X["cols"])
        if n_features and cols != n_features:
            raise ValueError(f"X has {cols} columns, expected {n_features}")
        data = np.ascontiguousarray(X["data"], dtype=np.float32).reshape(-1)
        if data.size != rows * cols:
            raise ValueError("matrix data length does not match rows * cols")
        return rows, cols, data
    arr = np.ascontiguousarray(X, dtype=np.float32)
    if arr.ndim == 1:
        if not n_features:
            raise ValueError("n_features is required for flat typed-array input")
        if arr.size % int(n_features) != 0:
            raise ValueError("flat X length is not divisible by n_features")
        arr = arr.reshape(-1, int(n_features))
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {arr.ndim}")
    if n_features and arr.shape[1] != int(n_features):
        raise ValueError(f"X has {arr.shape[1]} columns, expected {n_features}")
    if not np.isfinite(arr).all():
        raise ValueError("X must contain only finite values")
    return int(arr.shape[0]), int(arr.shape[1]), arr.reshape(-1)


def normalize_y(y, rows: int):
    arr = np.ascontiguousarray(y, dtype=np.float32).reshape(-1)
    if arr.shape[0] != int(rows):
        raise ValueError(f"y length {arr.shape[0]} does not match X rows {rows}")
    if not np.isfinite(arr).all():
        raise ValueError("y must contain only finite values")
    return arr


def _clone(c: Candidate) -> Candidate:
    return Candidate(
        terms=c.terms,
        fa=c.fa.copy(),
        fb=c.fb.copy(),
        op=c.op.copy(),
        p0=c.p0.copy(),
        p1=c.p1.copy(),
        coef=c.coef.copy(),
        bias=float(c.bias),
        loss=float(c.loss),
        objective=float(c.objective),
        complexity=float(c.complexity),
        generation=int(c.generation),
    )


def _random_candidate(rng, n_features: int, terms: int, op_ids=None) -> Candidate:
    ops = op_ids or list(range(len(OPS)))
    c = Candidate(
        terms=terms,
        fa=np.zeros(terms, dtype=np.int32),
        fb=np.zeros(terms, dtype=np.int32),
        op=np.zeros(terms, dtype=np.int32),
        p0=np.zeros(terms, dtype=np.float32),
        p1=np.zeros(terms, dtype=np.float32),
        coef=np.zeros(terms, dtype=np.float32),
        bias=_rand_range(rng, -0.5, 0.5),
        complexity=terms + 1,
    )
    for t in range(terms):
        c.fa[t] = _rand_int(rng, n_features)
        c.fb[t] = _rand_int(rng, n_features)
        c.op[t] = ops[_rand_int(rng, len(ops))]
        c.p0[t] = _rand_range(rng, -2, 2)
        c.p1[t] = _rand_range(rng, -1, 1)
        c.coef[t] = _rand_range(rng, -1, 1)
    return c


def _complexity(c: Candidate) -> float:
    value = 1.0
    for t in range(c.terms):
        value += 3.0 if int(c.op[t]) >= 4 else 2.0
        if abs(float(c.p0[t]) - 1.0) > 1e-6:
            value += 1.0
        if abs(float(c.p1[t])) > 1e-6:
            value += 1.0
    return value


def _candidate_key(c: Candidate) -> tuple:
    parts = []
    for t in range(c.terms):
        parts.extend([
            int(c.fa[t]),
            int(c.fb[t]),
            int(c.op[t]),
            int(round(float(c.p0[t]) * 1000)),
            int(round(float(c.p1[t]) * 1000)),
        ])
    return tuple(parts)


def _evaluation_key(c: Candidate) -> tuple:
    parts = []
    for t in range(c.terms):
        parts.extend([
            int(c.fa[t]),
            int(c.fb[t]),
            int(c.op[t]),
            float(np.float32(c.p0[t])).hex(),
            float(np.float32(c.p1[t])).hex(),
        ])
    return tuple(parts)


def _apply_cached_evaluation(c: Candidate, cached):
    c.bias = float(cached["bias"])
    c.coef[:] = cached["coef"]
    c.loss = float(cached["loss"])
    c.objective = float(cached["objective"])
    c.complexity = float(cached["complexity"])


def _remember_evaluation(cache, c: Candidate, max_size: int):
    if cache is None or max_size <= 0 or not math.isfinite(float(c.objective)):
        return
    cache[_evaluation_key(c)] = {
        "bias": float(c.bias),
        "coef": c.coef.copy(),
        "loss": float(c.loss),
        "objective": float(c.objective),
        "complexity": float(c.complexity),
    }
    while len(cache) > max_size:
        cache.pop(next(iter(cache)))


def _mutate(parent: Candidate, rng, n_features: int, params: dict[str, Any]) -> Candidate:
    op_ids = params.get("_opIds") or list(range(len(OPS)))
    c = _clone(parent)
    t = _rand_int(rng, c.terms)
    r = rng()
    if r < 0.20:
        c.fa[t] = _rand_int(rng, n_features)
    elif r < 0.40:
        c.fb[t] = _rand_int(rng, n_features)
    elif r < 0.58:
        c.op[t] = op_ids[_rand_int(rng, len(op_ids))]
    elif r < 0.78:
        c.p0[t] = max(-4.0, min(4.0, float(c.p0[t]) + _rand_range(rng, -0.4, 0.4)))
    elif r < 0.94:
        c.p1[t] = max(-4.0, min(4.0, float(c.p1[t]) + _rand_range(rng, -0.4, 0.4)))
    else:
        q = _rand_int(rng, c.terms)
        c.fa[q] = _rand_int(rng, n_features)
        c.fb[q] = _rand_int(rng, n_features)
        c.op[q] = op_ids[_rand_int(rng, len(op_ids))]
        c.p0[q] = _rand_range(rng, -2, 2)
        c.p1[q] = _rand_range(rng, -1, 1)
    if rng() < float(params.get("constantMutationRate", 0.25)):
        q = _rand_int(rng, c.terms)
        c.p0[q] = max(-4.0, min(4.0, float(c.p0[q]) + _rand_range(rng, -0.2, 0.2)))
        c.p1[q] = max(-4.0, min(4.0, float(c.p1[q]) + _rand_range(rng, -0.2, 0.2)))
    c.loss = math.inf
    c.objective = math.inf
    return c


def _crossover(a: Candidate, b: Candidate, rng) -> Candidate:
    c = _clone(a)
    for t in range(c.terms):
        if rng() < 0.5:
            c.fa[t] = b.fa[t]
            c.fb[t] = b.fb[t]
            c.op[t] = b.op[t]
            c.p0[t] = b.p0[t]
            c.p1[t] = b.p1[t]
    c.loss = math.inf
    c.objective = math.inf
    return c


def _protected_div(a: float, b: float) -> float:
    eps = 1e-6
    den = eps if abs(b) < eps and b >= 0 else (-eps if abs(b) < eps else b)
    return a / den


def _apply_op(op: int, a: float, b: float, p0: float, p1: float) -> float:
    z = a * p0 + p1
    if op == 0:
        return a + b
    if op == 1:
        return a - b
    if op == 2:
        return a * b
    if op == 3:
        return _protected_div(a, b)
    if op == 4:
        return math.sin(z)
    if op == 5:
        return math.cos(z)
    if op == 6:
        return math.tanh(z)
    if op == 7:
        return math.log(abs(z) + 1e-6)
    if op == 8:
        return math.sqrt(abs(z))
    return math.exp(max(-6.0, min(6.0, z)))


def _r2(y, pred):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    mean = float(np.mean(y)) if y.size else 0.0
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - mean) ** 2))
    return 1.0 if ss_tot <= 1e-24 and ss_res <= 1e-24 else (0.0 if ss_tot <= 1e-24 else 1.0 - ss_res / ss_tot)


def _solve_linear(a, b, n: int, ridge: float):
    mat = np.asarray(a, dtype=np.float64).reshape(n, n).copy()
    rhs = np.asarray(b, dtype=np.float64).reshape(n).copy()
    mat.flat[:: n + 1] += float(ridge)
    try:
        return np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.lstsq(mat, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None


def _serializable_params(params):
    return {
        k: v
        for k, v in (params or {}).items()
        if k not in {"polygrad", "polygradRuntime", "wasm", "_opIds"}
    }


class PgFamilyRegressorEngine:
    def __init__(self, params: Optional[dict[str, Any]] = None):
        self.params = {
            "population": 128,
            "generations": 20,
            "terms": 6,
            "eliteCount": 8,
            "tournamentSize": 4,
            "frontierSize": 16,
            "complexityPenalty": 1e-5,
            "ridge": 1e-8,
            "seed": 42,
            "operatorSet": "full",
            "stackSummaries": "auto",
            "evalCacheSize": 8192,
            **(params or {}),
        }
        self._op_ids = resolve_operator_ids(self.params)
        self.params["_opIds"] = self._op_ids
        self.pg = None
        self.best: Optional[Candidate] = None
        self.archive: list[Candidate] = []
        self.n_features = 0
        self.feature_names: list[str] = []
        self.fitted = False
        self.stats: Optional[dict[str, Any]] = None
        self._owns_pg = False

    def _load_polygrad(self):
        if self.pg is None:
            supplied = self.params.get("polygradRuntime", self.params.get("polygrad"))
            if supplied is not None and hasattr(supplied, "Tensor"):
                self.pg = supplied
            elif isinstance(supplied, dict):
                import polygrad  # noqa: PLC0415

                self.pg = polygrad.create(device=supplied.get("device", "auto"))
                self._owns_pg = True
            else:
                import polygrad  # noqa: PLC0415

                self.pg = polygrad
        return self.pg

    def _tensor(self, data, dtype="float32"):
        pg = self._load_polygrad()
        return pg.Tensor(data, dtype=dtype)

    def _make_inputs(self, population: list[Candidate], rows: int, cols: int, data, y=None, coeffs=True, cache=None):
        pg = self._load_polygrad()
        inputs = {
            "cols": cols,
            "x": cache["x"] if cache and "x" in cache else self._tensor(data),
            "y": cache["y"] if cache and "y" in cache else self._tensor(np.zeros(rows, dtype=np.float32) if y is None else y),
            "bias": self._tensor(np.asarray([c.bias for c in population], dtype=np.float32)),
            "sel_a": [],
            "sel_b": [],
            "p0": [],
            "p1": [],
            "coef": [],
            "mask": [],
            "op_ids": self._op_ids,
            "Tensor": pg.Tensor,
        }
        for t in range(int(self.params["terms"])):
            sel_a = np.zeros((len(population), cols), dtype=np.float32)
            sel_b = np.zeros((len(population), cols), dtype=np.float32)
            p0 = np.zeros(len(population), dtype=np.float32)
            p1 = np.zeros(len(population), dtype=np.float32)
            coef = np.zeros(len(population), dtype=np.float32)
            masks = [np.zeros(len(population), dtype=np.float32) for _ in self._op_ids]
            for i, cand in enumerate(population):
                p0[i] = cand.p0[t]
                p1[i] = cand.p1[t]
                coef[i] = cand.coef[t] if coeffs else 0.0
                try:
                    pos = self._op_ids.index(int(cand.op[t]))
                except ValueError as exc:
                    raise ValueError(f"candidate uses operator outside configured pg-family operators: {OPS[int(cand.op[t])] if 0 <= int(cand.op[t]) < len(OPS) else cand.op[t]}") from exc
                masks[pos][i] = 1.0
                sel_a[i, int(cand.fa[t])] = 1.0
                sel_b[i, int(cand.fb[t])] = 1.0
            inputs["sel_a"].append(self._tensor(sel_a.reshape(-1)))
            inputs["sel_b"].append(self._tensor(sel_b.reshape(-1)))
            inputs["p0"].append(self._tensor(p0))
            inputs["p1"].append(self._tensor(p1))
            inputs["coef"].append(self._tensor(coef))
            inputs["mask"].append([self._tensor(mask) for mask in masks])
        return inputs

    def _branch_for_op(self, op, a, b, z):
        if op == 0:
            return a + b
        if op == 1:
            return a - b
        if op == 2:
            return a * b
        if op == 3:
            safe_den = (b.abs() < 1e-6).where((b >= 0).where(1e-6, -1e-6), b)
            return a / safe_den
        if op == 4:
            return z.sin()
        if op == 5:
            return z.cos()
        if op == 6:
            return z.tanh()
        if op == 7:
            return (z.abs() + 1e-6).log()
        if op == 8:
            return z.abs().sqrt()
        return z.clamp(-6, 6).exp()

    def _term_outputs(self, inputs, rows: int, candidates: int):
        Tensor = inputs["Tensor"]
        x_t = inputs["x"].reshape(rows, int(inputs["cols"])).transpose()
        out = []
        for t in range(int(self.params["terms"])):
            a = inputs["sel_a"][t].reshape(candidates, int(inputs["cols"])).dot(x_t)
            b = inputs["sel_b"][t].reshape(candidates, int(inputs["cols"])).dot(x_t)
            p0 = inputs["p0"][t].reshape(candidates, 1)
            p1 = inputs["p1"][t].reshape(candidates, 1)
            z = a * p0 + p1
            term = Tensor.zeros(candidates, rows, dtype="float32")
            for pos, op in enumerate(inputs["op_ids"]):
                term = term + inputs["mask"][t][pos].reshape(candidates, 1) * self._branch_for_op(op, a, b, z)
            out.append(term)
        return out

    def _summary_arrays(self, population, rows, cols, data, y, cache=None):
        inputs = self._make_inputs(population, rows, cols, data, y, coeffs=False, cache=cache)
        terms = self._term_outputs(inputs, rows, len(population))
        y_t = inputs["y"].reshape(1, rows)
        outs = []
        for term in terms:
            outs.append(term.sum(axis=1))
        for term in terms:
            outs.append((term * y_t).sum(axis=1))
        for i, left in enumerate(terms):
            for right in terms[i:]:
                outs.append((left * right).sum(axis=1))
        Tensor = inputs["Tensor"]
        if self.params.get("stackSummaries") is True:
            stacked = Tensor.stack(*outs, dim=0)
            arr = np.asarray(stacked.numpy(), dtype=np.float32).reshape(len(outs), len(population))
            return [arr[i].copy() for i in range(arr.shape[0])]
        return [np.asarray(arr, dtype=np.float32).reshape(-1) for arr in Tensor.numpy_many(*outs)]

    def _fit_coefficients(self, population, rows, cols, data, y, cache=None):
        arrays = self._summary_arrays(population, rows, cols, data, y, cache=cache)
        n = int(self.params["terms"]) + 1
        sum_y = float(np.sum(y))
        y_ty = float(np.sum(y * y))
        y_offset = int(self.params["terms"])
        cross_offset = int(self.params["terms"]) * 2
        for ci, cand in enumerate(population):
            mat = np.zeros((n, n), dtype=np.float64)
            rhs = np.zeros(n, dtype=np.float64)
            mat[0, 0] = rows
            rhs[0] = sum_y
            for t in range(int(self.params["terms"])):
                st = float(arrays[t][ci])
                mat[0, t + 1] = st
                mat[t + 1, 0] = st
                rhs[t + 1] = float(arrays[y_offset + t][ci])
            k = cross_offset
            for t in range(int(self.params["terms"])):
                for s in range(t, int(self.params["terms"])):
                    value = float(arrays[k][ci])
                    k += 1
                    mat[t + 1, s + 1] = value
                    mat[s + 1, t + 1] = value
            sol = _solve_linear(mat, rhs, n, float(self.params.get("ridge", 1e-8)))
            cand.complexity = _complexity(cand)
            if sol is None or not np.isfinite(sol).all():
                cand.loss = math.inf
                cand.objective = math.inf
                continue
            cand.bias = float(sol[0])
            cand.coef[:] = np.asarray(sol[1:], dtype=np.float32)
            quad = float(sol @ mat @ sol)
            lin = float(sol @ rhs)
            cand.loss = max(0.0, (quad - 2.0 * lin + y_ty) / max(1, rows))
            cand.objective = cand.loss + float(self.params["complexityPenalty"]) * cand.complexity

    def _fit_needed(self, population, rows, cols, data, y, cache=None, eval_cache=None):
        pending = []
        cache_hits = 0
        max_size = int(self.params.get("evalCacheSize", 8192))
        for cand in population:
            if math.isfinite(float(cand.objective)) and math.isfinite(float(cand.loss)):
                _remember_evaluation(eval_cache, cand, max_size)
                continue
            cached = eval_cache.get(_evaluation_key(cand)) if eval_cache is not None else None
            if cached is not None:
                _apply_cached_evaluation(cand, cached)
                cache_hits += 1
            else:
                pending.append(cand)
        if pending:
            self._fit_coefficients(pending, rows, cols, data, y, cache=cache)
            for cand in pending:
                _remember_evaluation(eval_cache, cand, max_size)
        return {"fitCount": len(pending), "cacheHits": cache_hits}

    def _score_cpu(self, population, rows, cols, data, y):
        for cand in population:
            pred = self._predict_cpu(rows, cols, data, cand)
            cand.loss = float(np.mean((pred.astype(np.float64) - y.astype(np.float64)) ** 2))
            cand.complexity = _complexity(cand)
            cand.objective = cand.loss + float(self.params["complexityPenalty"]) * cand.complexity

    def _archive(self, population, generation: int):
        by_key = {_candidate_key(old): old for old in self.archive}
        for cand in population:
            item = _clone(cand)
            item.generation = generation
            key = _candidate_key(item)
            prev = by_key.get(key)
            if prev is None or item.objective < prev.objective:
                by_key[key] = item
        self.archive = sorted(by_key.values(), key=lambda c: (c.objective, c.loss, c.complexity))[: int(self.params["frontierSize"])]
        self.best = _clone(self.archive[0])

    def _select(self, population, rng):
        best = None
        for _ in range(int(self.params["tournamentSize"])):
            cand = population[_rand_int(rng, len(population))]
            if best is None or cand.objective < best.objective:
                best = cand
        return best

    def fit(self, X, y):
        rows, cols, data = normalize_x(X)
        target = normalize_y(y, rows)
        self._load_polygrad()
        self.n_features = cols
        self.feature_names = [f"x{i}" for i in range(cols)]
        cache = {"x": self._tensor(data), "y": self._tensor(target)}
        eval_cache = None if int(self.params.get("evalCacheSize", 8192)) == 0 else {}
        rng = _lcg(int(self.params.get("seed", 42)))
        population = [
            _random_candidate(rng, cols, int(self.params["terms"]), self._op_ids)
            for _ in range(int(self.params["population"]))
        ]
        timings = []
        for gen in range(int(self.params["generations"])):
            fit_report = self._fit_needed(population, rows, cols, data, target, cache=cache, eval_cache=eval_cache)
            if self.params.get("scoreMode") not in ("summary", None):
                self._score_cpu(population, rows, cols, data, target)
            population.sort(key=lambda c: c.objective)
            self._archive(population, gen)
            next_pop = [_clone(c) for c in population[: int(self.params["eliteCount"])]]
            seen = {_candidate_key(c) for c in next_pop}
            while len(next_pop) < int(self.params["population"]):
                if rng() < 0.20:
                    child = _random_candidate(rng, cols, int(self.params["terms"]), self._op_ids)
                elif rng() < 0.55:
                    child = _crossover(self._select(population, rng), self._select(population, rng), rng)
                else:
                    child = _mutate(self._select(population, rng), rng, cols, self.params)
                key = _candidate_key(child)
                if key not in seen or len(next_pop) > int(self.params["population"]) * 0.95:
                    seen.add(key)
                    next_pop.append(child)
            timings.append({
                "generation": gen,
                "fitCount": fit_report["fitCount"],
                "cacheHits": fit_report["cacheHits"],
                "bestLoss": float(self.best.loss),
                "bestObjective": float(self.best.objective),
            })
            population = next_pop
        self._fit_needed(self.archive, rows, cols, data, target, cache=cache, eval_cache=eval_cache)
        self.archive.sort(key=lambda c: c.objective)
        self.best = _clone(self.archive[0])
        self.fitted = True
        self.stats = {"trainLoss": float(self.best.loss), "timings": timings}
        return self

    def _predict_cpu(self, rows: int, cols: int, data, cand: Optional[Candidate] = None):
        cand = cand or self.best
        if cand is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(data, dtype=np.float32).reshape(rows, cols)
        out = np.zeros(rows, dtype=np.float32)
        for r in range(rows):
            pred = float(cand.bias)
            for t in range(cand.terms):
                a = float(X[r, int(cand.fa[t])])
                b = float(X[r, int(cand.fb[t])])
                pred += float(cand.coef[t]) * _apply_op(int(cand.op[t]), a, b, float(cand.p0[t]), float(cand.p1[t]))
            out[r] = pred
        return out

    def predict(self, X):
        if not self.fitted or self.best is None:
            raise RuntimeError("model is not fitted")
        rows, cols, data = normalize_x(X, self.n_features)
        return self._predict_cpu(rows, cols, data).astype(np.float64)

    def score(self, X, y):
        rows, _, _ = normalize_x(X, self.n_features)
        target = normalize_y(y, rows)
        return float(_r2(target, self.predict(X)))

    def formula(self, format="json", index=0):
        if self.best is None:
            raise RuntimeError("model is not fitted")
        cand = self.best if int(index) == 0 else self.archive[int(index)]
        text = _formula_text(cand, self.feature_names)
        if format == "text":
            return text
        return {
            "kind": "sym.pg-family.formula@1",
            "text": text,
            "terms": [
                {
                    "coefficient": float(cand.coef[t]),
                    "op": OPS[int(cand.op[t])],
                    "featureA": int(cand.fa[t]),
                    "featureB": int(cand.fb[t]),
                    "p0": float(cand.p0[t]),
                    "p1": float(cand.p1[t]),
                }
                for t in range(cand.terms)
            ],
            "bias": float(cand.bias),
            "loss": float(cand.loss),
            "objective": float(cand.objective),
            "complexity": float(cand.complexity),
        }

    def frontier(self, format=None):
        out = []
        for cand in self.archive:
            item = _candidate_to_plain(cand)
            if format == "text":
                item["text"] = _formula_text(cand, self.feature_names)
            out.append(item)
        return out

    def to_state(self):
        if not self.fitted or self.best is None:
            raise RuntimeError("model is not fitted")
        return {
            "kind": "wlearn.sym.pg-family.regressor@1",
            "params": _serializable_params(self.params),
            "nFeatures": int(self.n_features),
            "featureNames": self.feature_names,
            "fitted": bool(self.fitted),
            "best": _candidate_to_plain(self.best),
            "archive": [_candidate_to_plain(c) for c in self.archive],
            "stats": self.stats,
        }

    @classmethod
    def from_state(cls, payload, params=None):
        if not payload or payload.get("kind") != "wlearn.sym.pg-family.regressor@1":
            raise ValueError("not a wlearn sym pg-family regressor state")
        obj = cls({**(payload.get("params") or {}), **(params or {})})
        obj.n_features = int(payload.get("nFeatures") or 0)
        obj.feature_names = list(payload.get("featureNames") or [f"x{i}" for i in range(obj.n_features)])
        obj.fitted = bool(payload.get("fitted"))
        obj.best = _candidate_from_plain(payload["best"]) if payload.get("best") else None
        obj.archive = [_candidate_from_plain(item) for item in payload.get("archive") or []]
        obj.stats = payload.get("stats")
        return obj

    def dispose(self):
        if self._owns_pg and self.pg is not None and hasattr(self.pg, "dispose"):
            self.pg.dispose()
        self._owns_pg = False
        self.pg = None


class PgFamilyClassifierEngine:
    def __init__(self, params: Optional[dict[str, Any]] = None):
        self.params = dict(params or {})
        self.classes: list[float] = []
        self.models: list[PgFamilyRegressorEngine] = []
        self.n_features = 0
        self.fitted = False
        self.stats = None

    def fit(self, X, y):
        rows, cols, data = normalize_x(X)
        target = normalize_y(y, rows)
        self.classes = [float(v) for v in (self.params.get("classes") or sorted(np.unique(target).tolist()))]
        if len(self.classes) < 2:
            raise ValueError("classification requires at least two classes")
        self.n_features = cols
        self.models = []
        if len(self.classes) == 2:
            bin_y = np.where(target == self.classes[1], 1.0, -1.0).astype(np.float32)
            model = PgFamilyRegressorEngine({k: v for k, v in self.params.items() if k != "classes"})
            model.fit({"rows": rows, "cols": cols, "data": data}, bin_y)
            self.models.append(model)
        else:
            for cls in self.classes:
                one = np.where(target == cls, 1.0, -1.0).astype(np.float32)
                model = PgFamilyRegressorEngine({k: v for k, v in self.params.items() if k != "classes"})
                model.fit({"rows": rows, "cols": cols, "data": data}, one)
                self.models.append(model)
        self.fitted = True
        self.stats = {"runtimes": [m.stats for m in self.models]}
        return self

    def _ensure_fitted(self):
        if not self.fitted or not self.models:
            raise RuntimeError("model is not fitted")

    def decision_function(self, X):
        self._ensure_fitted()
        rows, _, _ = normalize_x(X, self.n_features)
        if len(self.classes) == 2:
            return self.models[0].predict(X).reshape(rows)
        out = np.zeros((rows, len(self.classes)), dtype=np.float64)
        for i, model in enumerate(self.models):
            out[:, i] = model.predict(X)
        return out

    def predict_proba(self, X):
        self._ensure_fitted()
        scores = self.decision_function(X)
        if len(self.classes) == 2:
            s = np.clip(scores.reshape(-1), -60, 60)
            p1 = 1.0 / (1.0 + np.exp(-s))
            return np.column_stack([1.0 - p1, p1])
        scores = np.asarray(scores, dtype=np.float64)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(np.clip(scores, -60, 60))
        denom = np.sum(exp, axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        return exp / denom

    def predict(self, X):
        self._ensure_fitted()
        rows, _, _ = normalize_x(X, self.n_features)
        if len(self.classes) == 2:
            scores = self.decision_function(X).reshape(rows)
            return np.asarray([self.classes[1] if v >= 0 else self.classes[0] for v in scores], dtype=np.float64)
        proba = self.predict_proba(X)
        return np.asarray([self.classes[int(i)] for i in np.argmax(proba, axis=1)], dtype=np.float64)

    def score(self, X, y):
        rows, _, _ = normalize_x(X, self.n_features)
        target = normalize_y(y, rows)
        return float(np.mean(self.predict(X) == target))

    def formula(self, format="json", index=0):
        self._ensure_fitted()
        model = self.models[0] if len(self.classes) == 2 else self.models[int(index)]
        item = model.formula(format=format)
        if format == "text":
            return item
        item["classLabel"] = self.classes[1] if len(self.classes) == 2 else self.classes[int(index)]
        item["role"] = "binary-margin" if len(self.classes) == 2 else "one-vs-rest-margin"
        return item

    def frontier(self, format=None):
        self._ensure_fitted()
        out = []
        for idx, model in enumerate(self.models):
            label = self.classes[1] if len(self.classes) == 2 else self.classes[idx]
            class_index = 1 if len(self.classes) == 2 else idx
            for item in model.frontier(format=format):
                item["classIndex"] = class_index
                item["classLabel"] = label
                out.append(item)
        return out

    def to_state(self):
        self._ensure_fitted()
        return {
            "kind": "wlearn.sym.pg-family.classifier@1",
            "params": _serializable_params(self.params),
            "classes": self.classes,
            "nFeatures": int(self.n_features),
            "fitted": bool(self.fitted),
            "models": [model.to_state() for model in self.models],
            "stats": self.stats,
        }

    @classmethod
    def from_state(cls, payload, params=None):
        if not payload or payload.get("kind") != "wlearn.sym.pg-family.classifier@1":
            raise ValueError("not a wlearn sym pg-family classifier state")
        obj = cls({**(payload.get("params") or {}), **(params or {})})
        obj.classes = [float(v) for v in payload.get("classes") or []]
        obj.n_features = int(payload.get("nFeatures") or 0)
        obj.fitted = bool(payload.get("fitted"))
        obj.models = [PgFamilyRegressorEngine.from_state(state) for state in payload.get("models") or []]
        obj.stats = payload.get("stats")
        return obj

    def dispose(self):
        for model in self.models:
            model.dispose()
        self.models = []
        self.fitted = False

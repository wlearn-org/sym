"""C family binding: buffer transport and existing WLRN payload adaptation."""

import ctypes
import numpy as np
from ._ffi import get_lib, last_error, _DP
from ._family_format import (
    OPS,
    resolve_operator_ids,
    _formula_text,
    _candidate_from_plain,
)


def _config(p):
    def get(a, b, default):
        return p.get(a, p.get(b, default))

    if p.get("loss", "mse") not in ("mse", 0):
        raise ValueError(
            "family currently supports the MSE readout, including classification margins"
        )
    for k in (
        "maxNodes",
        "max_nodes",
        "maxDepth",
        "max_depth",
        "topK",
        "top_k",
        "broodSize",
        "brood_size",
        "rowSampleSize",
        "row_sample_size",
        "complexityHofSize",
        "complexity_hof_size",
        "finalSelector",
        "final_selector",
        "hierarchical",
        "migrationInterval",
        "migration_interval",
        "migrationCount",
        "migration_count",
        "localRefineInterval",
        "local_refine_interval",
        "localRefineCount",
        "local_refine_count",
        "scoreMode",
        "score_mode",
        "jit",
        "gatherMode",
        "gather_mode",
        "polygradRuntime",
        "polygrad",
        "tournamentSize",
        "tournament_size",
        "constantRate",
        "constant_rate",
        "constMin",
        "const_min",
        "constMax",
        "const_max",
        "huberDelta",
        "huber_delta",
        "warmupGenerations",
        "warmup_generations",
        "warmupMinNodes",
        "warmup_min_nodes",
        "complexityBucketWidth",
        "complexity_bucket_width",
    ):
        if p.get(k) is not None:
            raise ValueError(f"{k} is not supported by C family search")
    mask = sum(1 << op for op in resolve_operator_ids(p))
    values = [
        p.get("population", 128),
        p.get("generations", 20),
        p.get("terms", 6),
        get("eliteCount", "elite_count", 8),
        p.get("islands", 1),
        get("frontierSize", "frontier_size", 16),
        mask,
        p.get("seed", 42),
        get("polishPasses", "polish_passes", 0),
        get("earlyStopRounds", "early_stop_rounds", 0),
        get("validationFraction", "validation_fraction", 0),
        get("complexityPenalty", "complexity_penalty", 1e-5),
        get("immigrantRate", "immigrant_rate", 0),
        get("mutationRate", "mutation_rate", 0.35),
        get("crossoverRate", "crossover_rate", 0.55),
        p.get("ridge", 1e-8),
        p.get("tol", 1e-12),
    ]
    if any(
        isinstance(v, (bool, str)) or not isinstance(v, (int, float, np.number))
        for v in values
    ):
        raise ValueError("family parameters must be finite numbers")
    return np.asarray(values, dtype=np.float64)


def _matrix(X, cols):
    if isinstance(X, dict):
        X = np.asarray(X["data"]).reshape(X["rows"], X["cols"])
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, cols)
    if X.ndim != 2 or X.shape[1] != cols:
        raise ValueError("family matrix shape mismatch")
    return X


def _packed(c):
    n = c.get("terms") if isinstance(c, dict) else None
    if type(n) is not int or not 1 <= n <= 32:
        raise ValueError("invalid family term count")
    keys = ("fa", "fb", "op", "p0", "p1", "coef")
    if any(not isinstance(c.get(k), list) or len(c[k]) != n for k in keys):
        raise ValueError("invalid family term arrays")
    data = [c[k][t] for t in range(n) for k in keys]
    data += [
        c.get("bias"),
        c.get("loss"),
        c.get("validLoss", c.get("loss")),
        c.get("complexity"),
        c.get("objective"),
        c.get("generation", 0),
    ]
    if any(isinstance(v, (bool, str)) or not isinstance(v, (int, float)) for v in data):
        raise ValueError("invalid family numeric value")
    return np.asarray(data, dtype=np.float64)


class CFamilyEngine:
    def __init__(self, params=None, classes=None):
        self.params = dict(params or {})
        self.classes = list(classes or [])
        self.handle = None
        self.n_features = self.terms = 0
        self.artifact_version = 1

    def fit(self, X, y):
        lib = get_lib()
        options = _config(self.params)
        X = np.ascontiguousarray(X, dtype=np.float64)
        if self.classes:
            index = {v: i for i, v in enumerate(self.classes)}
            y = [index[v] for v in y]
        target = np.ascontiguousarray(y, dtype=np.float64)
        handle = lib.wl_sym_family_fit(
            X.ctypes.data_as(_DP),
            X.shape[0],
            X.shape[1],
            target.ctypes.data_as(_DP),
            bool(self.classes),
            len(self.classes),
            options.ctypes.data_as(_DP),
            options.size,
        )
        if not handle:
            raise ValueError(last_error(lib))
        self.dispose()
        self.handle = handle
        self.n_features = X.shape[1]
        self.terms = int(options[2])
        self.artifact_version = 2
        return self

    def _predict(self, X, mode):
        X = _matrix(X, self.n_features)
        heads = len(self.classes) if len(self.classes) > 2 else 1
        cols = len(self.classes) if mode == 2 else heads if mode == 1 else 1
        out = np.empty((X.shape[0], cols), dtype=np.float64)
        lib = get_lib()
        if lib.sym_family_predict(
            self.handle,
            X.ctypes.data_as(_DP),
            X.shape[0],
            X.shape[1],
            mode,
            out.ctypes.data_as(_DP),
        ):
            raise ValueError(last_error(lib))
        return out if cols > 1 else out[:, 0]

    def predict(self, X, backend=None, **kwargs):
        if backend == "polygrad":
            raise ValueError(
                "Polygrad execution for C family models is not implemented yet"
            )
        out = self._predict(X, 0)
        return np.asarray(self.classes)[out.astype(int)] if self.classes else out

    def decision_function(self, X):
        return self._predict(X, 1)

    def predict_proba(self, X):
        return self._predict(X, 2)

    def score(self, X, y):
        pred = self.predict(X)
        y = np.asarray(y)
        if y.shape != pred.shape:
            raise ValueError("target shape mismatch")
        if self.classes:
            return float(np.mean(pred == y))
        residual = float(np.sum((pred - y) ** 2))
        total = float(np.sum((y - y.mean()) ** 2))
        return 1 - residual / total if total else float(residual == 0)

    def _candidate(self, head, index):
        data = np.zeros(6 * self.terms + 6, dtype=np.float64)
        lib = get_lib()
        if (
            lib.sym_family_export(
                self.handle, head, index, data.ctypes.data_as(_DP), data.size
            )
            < 0
        ):
            raise ValueError(last_error(lib))
        c = {"terms": self.terms}
        for j, key in enumerate(("fa", "fb", "op", "p0", "p1", "coef")):
            c[key] = data[: 6 * self.terms].reshape(-1, 6)[:, j].tolist()
            if j < 3:
                c[key] = [int(v) for v in c[key]]
        c.update(
            zip(
                ("bias", "loss", "validLoss", "complexity", "objective", "generation"),
                data[6 * self.terms :].tolist(),
            )
        )
        c["generation"] = int(c["generation"])
        return c

    def formula(self, format="json", index=0):
        head = index if len(self.classes) > 2 else 0
        c = self._candidate(head, -1)
        text = _formula_text(
            _candidate_from_plain(c), [f"x{i}" for i in range(self.n_features)]
        )
        if format == "text":
            return text
        out = {
            k: c[k] for k in ("bias", "loss", "validLoss", "objective", "complexity")
        }
        out.update(
            kind="sym.pg-family.formula@1",
            text=text,
            terms=[
                dict(
                    coefficient=c["coef"][i],
                    op=OPS[op],
                    featureA=c["fa"][i],
                    featureB=c["fb"][i],
                    p0=c["p0"][i],
                    p1=c["p1"][i],
                )
                for i, op in enumerate(c["op"])
            ],
        )
        if self.classes:
            out.update(
                classLabel=self.classes[1 if len(self.classes) == 2 else head],
                role="binary-margin"
                if len(self.classes) == 2
                else "one-vs-rest-margin",
            )
        return out

    def frontier(self, format=None):
        out = []
        heads = len(self.classes) if len(self.classes) > 2 else 1
        for h in range(heads):
            for i in range(get_lib().sym_family_archive_size(self.handle, h)):
                c = self._candidate(h, i)
                if format == "text":
                    c["text"] = _formula_text(
                        _candidate_from_plain(c),
                        [f"x{j}" for j in range(self.n_features)],
                    )
                if self.classes:
                    c.update(
                        classIndex=1 if len(self.classes) == 2 else h,
                        classLabel=self.classes[1 if len(self.classes) == 2 else h],
                    )
                out.append(c)
        return out

    def to_state(self):
        params = {
            k: v
            for k, v in self.params.items()
            if k not in ("polygrad", "wasm", "polygradRuntime")
        }
        heads = len(self.classes) if len(self.classes) > 2 else 1
        models = [
            dict(
                kind="wlearn.sym.pg-family.regressor@1",
                params=params,
                nFeatures=self.n_features,
                fitted=True,
                featureNames=[f"x{i}" for i in range(self.n_features)],
                best=self._candidate(h, -1),
                archive=[
                    self._candidate(h, i)
                    for i in range(get_lib().sym_family_archive_size(self.handle, h))
                ],
                stats=None,
            )
            for h in range(heads)
        ]
        return (
            dict(
                kind="wlearn.sym.pg-family.classifier@1",
                params=params,
                nFeatures=self.n_features,
                fitted=True,
                classes=self.classes,
                models=models,
                stats=None,
            )
            if self.classes
            else models[0]
        )

    def to_bytes(self):
        lib = get_lib()
        pointer = ctypes.c_void_p()
        size = ctypes.c_int()
        if lib.sym_family_save(self.handle, ctypes.byref(pointer), ctypes.byref(size)):
            raise ValueError(last_error(lib))
        try:
            return ctypes.string_at(pointer, size.value)
        finally:
            lib.wl_sym_free_buffer(pointer)

    @classmethod
    def from_bytes(cls, data, params, classes):
        lib = get_lib()
        buf = ctypes.create_string_buffer(data)
        obj = cls(params, classes)
        obj.handle = lib.sym_family_load(buf, len(data))
        if not obj.handle:
            raise ValueError(last_error(lib))
        try:
            dims = (ctypes.c_int * 4)()
            if lib.sym_family_dimensions(obj.handle, dims):
                raise ValueError(last_error(lib))
            if (
                not isinstance(classes, list)
                or len(classes) != dims[1]
                or any(
                    type(v) not in (int, float) or not np.isfinite(v) for v in classes
                )
                or len(set(classes)) != len(classes)
            ):
                raise ValueError("family artifact classes mismatch")
            obj.n_features, obj.terms = dims[0], dims[2]
            obj.artifact_version = 2
            return obj
        except Exception:
            obj.dispose()
            raise

    @classmethod
    def from_state(cls, state, params=None):
        if not isinstance(state, dict) or not state.get("fitted"):
            raise ValueError("family payload is not fitted")
        classifier = state.get("kind") == "wlearn.sym.pg-family.classifier@1"
        if not classifier and state.get("kind") != "wlearn.sym.pg-family.regressor@1":
            raise ValueError("invalid family payload kind")
        classes = state.get("classes") if classifier else []
        if (
            not isinstance(classes, list)
            or (classifier and not 2 <= len(classes) <= 128)
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValueError("invalid family classes")
        models = state.get("models") if classifier else [state]
        heads = len(classes) if len(classes) > 2 else 1
        if (
            not isinstance(models, list)
            or len(models) != heads
            or any(
                not isinstance(m, dict)
                or m.get("kind") != "wlearn.sym.pg-family.regressor@1"
                or not m.get("fitted")
                or m.get("nFeatures") != state.get("nFeatures")
                or not isinstance(m.get("archive"), list)
                for m in models
            )
        ):
            raise ValueError("invalid family heads")
        n = state.get("nFeatures")
        best = models[0].get("best")
        terms = best.get("terms") if isinstance(best, dict) else None
        frontier = max(1, *(len(m["archive"]) for m in models))
        if (
            type(n) is not int
            or not 1 <= n <= 2147483647
            or type(terms) is not int
            or not 1 <= terms <= 32
            or frontier > 128
        ):
            raise ValueError("invalid family dimensions")
        obj = cls({**state.get("params", {}), **(params or {})}, classes)
        lib = get_lib()
        obj.handle = lib.sym_family_new(n, len(classes), terms, frontier)
        if not obj.handle:
            raise ValueError(last_error(lib))
        obj.n_features, obj.terms = n, terms
        try:
            for h, model in enumerate(models):
                for i, c in enumerate([model["best"], *model["archive"]]):
                    if not isinstance(c, dict) or c.get("terms") != terms:
                        raise ValueError("family head term count mismatch")
                    data = _packed(c)
                    if lib.sym_family_import(
                        obj.handle, h, i - 1, data.ctypes.data_as(_DP), data.size
                    ):
                        raise ValueError(last_error(lib))
            return obj
        except Exception:
            obj.dispose()
            raise

    def dispose(self):
        if self.handle:
            get_lib().sym_family_free(self.handle)
            self.handle = None

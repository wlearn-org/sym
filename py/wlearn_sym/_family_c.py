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
        "stackSummaries",
        "evalCacheSize",
        "tensorDevice",
        "ownPolygradRuntime",
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
        "migrationInterval",
        "migration_interval",
        "migrationCount",
        "migration_count",
        "scoreMode",
        "score_mode",
        "jit",
        "gatherMode",
        "gather_mode",
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
    if "hierarchical" in p and not isinstance(p["hierarchical"], bool):
        raise ValueError("hierarchical must be boolean")
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
        get("polishBatchSize", "polish_batch_size", 0),
        int(p.get("hierarchical", False)),
        get("localRefineInterval", "local_refine_interval", 0),
        get("localRefineCount", "local_refine_count", 0),
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
    if X.ndim != 2 or X.shape[1] != cols or not np.isfinite(X).all():
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

    def predict_polygrad(self, X, polygrad=None):
        from ._family_refine import predict

        X = _matrix(X, self.n_features)
        heads = len(self.classes) if len(self.classes) > 2 else 1
        values = [
            predict(
                self._candidate(h, -1),
                X,
                polygrad
                or self.params.get("polygradRuntime")
                or self.params.get("polygrad"),
            )
            for h in range(heads)
        ]
        if not self.classes:
            return values[0]
        indices = (
            np.argmax(np.column_stack(values), axis=1)
            if heads > 1
            else (values[0] >= 0).astype(int)
        )
        return np.asarray(self.classes)[indices]

    def refine_polygrad(
        self,
        X,
        y,
        *,
        index=0,
        epochs=80,
        lr=0.001,
        optimizer="adam",
        tolerance=1e-10,
        polygrad=None,
    ):
        from ._family_refine import refine

        heads = len(self.classes) if len(self.classes) > 2 else 1
        if type(index) is not int or not 0 <= index < heads:
            raise ValueError("invalid family head index")
        X = _matrix(X, self.n_features)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("family target length mismatch")
        if self.classes:
            mapping = {v: i for i, v in enumerate(self.classes)}
            y = np.array([mapping[v] for v in y], dtype=np.float64)
        y = np.ascontiguousarray(y)
        lib = get_lib()
        fraction = self.params.get(
            "validationFraction", self.params.get("validation_fraction", 0)
        )
        seed = self.params.get("seed", 42)
        data = np.empty(2 * len(X) + 1)

        def ptr(a):
            return a.ctypes.data_as(_DP)

        if lib.sym_family_refine_data(
            self.handle, index, ptr(y), len(X), fraction, seed, ptr(data), data.size
        ):
            raise ValueError(last_error(lib))
        training = data[len(X) : 2 * len(X)].astype(bool)
        candidate = self._candidate(index, -1)
        updates, history = refine(
            candidate,
            X[training],
            data[: len(X)][training],
            epochs=epochs,
            lr=lr,
            optimizer=optimizer,
            ridge=self.params.get("ridge", 1e-8),
            polygrad=polygrad
            or self.params.get("polygradRuntime")
            or self.params.get("polygrad"),
        )
        report = np.empty(6)
        rc = lib.sym_family_refine_accept(
            self.handle,
            index,
            ptr(updates),
            updates.size,
            ptr(X),
            len(X),
            X.shape[1],
            ptr(y),
            fraction,
            seed,
            tolerance,
            ptr(report),
        )
        if rc < 0:
            raise ValueError(last_error(lib))
        return dict(
            status="ok",
            committed=bool(rc),
            history=history,
            head=index,
            beforeLoss=report[0],
            beforeValidLoss=report[1],
            beforeObjective=report[2],
            afterLoss=report[3],
            afterValidLoss=report[4],
            afterObjective=report[5],
        )

    def _fit_polygrad(self, xp, rows, cols, yp, task, classes, config, count):
        from ._family_scorer import FamilyScorer

        lib = get_lib()
        capacity = self.params.get(
            "batchSize", self.params.get("batch_size", min(int(config[0]), 512))
        )
        if type(capacity) is not int or not 1 <= capacity <= 512:
            raise ValueError("family batchSize must be an integer in [1,512]")
        search = lib.wl_sym_family_search_new(
            xp, rows, cols, yp, task, classes, config, count, capacity, min(rows, 4096)
        )
        if not search:
            raise ValueError(last_error(lib))
        terms = int(config[2])
        X = np.ctypeslib.as_array(xp, shape=(rows * cols,)).reshape(rows, cols).copy()
        descriptors = np.empty((capacity, terms, 5))
        shape = np.empty(8, dtype=np.int32)
        data = np.empty(2 * rows + 1)
        scorer = None
        head = -1
        actual = identity = 0

        def ptr(a):
            return a.ctypes.data_as(_DP)

        def solve(factors, stats):
            factors = np.ascontiguousarray(factors[:actual], dtype=np.float64)
            stats = np.ascontiguousarray(stats[:actual], dtype=np.float64)
            output = np.empty((actual, terms + 1))
            if lib.sym_family_search_solve(
                search,
                identity,
                ptr(factors),
                factors.size,
                ptr(stats),
                stats.size,
                factors.shape[-1],
                ptr(output),
                output.size,
            ):
                raise ValueError(last_error(lib))
            # Padding never creates additional candidates or consumes RNG.
            return np.concatenate(
                [output, np.repeat(output[:1], capacity - actual, axis=0)]
            )

        try:
            while True:
                actual = lib.wl_sym_family_search_propose(search)
                if actual < 0:
                    raise ValueError(last_error(lib))
                if actual == 0:
                    break
                identity = lib.wl_sym_family_batch_id(search)
                if lib.wl_sym_family_batch_shape(
                    search, shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 8
                ):
                    raise ValueError(last_error(lib))
                actual = int(shape[1])
                if lib.wl_sym_family_batch_descriptors(
                    search, ptr(descriptors), actual * terms * 5
                ):
                    raise ValueError(last_error(lib))
                descriptors[actual:] = descriptors[0]
                if head != shape[6]:
                    if scorer:
                        scorer.dispose()
                    head = int(shape[6])
                    if lib.sym_family_search_data(search, ptr(data), data.size):
                        raise ValueError(last_error(lib))
                    options = self.params.get("polygrad") or {}
                    device = (
                        options.get("device", "auto")
                        if isinstance(options, dict)
                        else "auto"
                    )
                    scorer = FamilyScorer(
                        X,
                        data[:rows],
                        1 - data[rows : 2 * rows],
                        descriptors,
                        solve=solve,
                        target_mean=data[-1],
                        hierarchical=self.params.get("hierarchical", False),
                        device=device,
                        runtime=self.params.get("polygradRuntime")
                        or (options if hasattr(options, "Tensor") else None),
                        dtype=self.params.get("scorerDtype", "float64"),
                    )
                result = scorer.score(descriptors)
                packed = np.ascontiguousarray(
                    np.concatenate(
                        [result["coefficients"][:actual], result["losses"][:actual]],
                        axis=1,
                    )
                )
                if lib.sym_family_search_accept_results(
                    search, identity, ptr(packed), packed.size
                ):
                    raise ValueError(last_error(lib))
            model = lib.sym_family_search_finish(search)
            if not model:
                raise ValueError(last_error(lib))
            return model
        finally:
            if scorer:
                scorer.dispose()
            lib.sym_family_search_free(search)

    def fit(self, X, y):
        lib = get_lib()
        options = _config(self.params)
        X = np.ascontiguousarray(X, dtype=np.float64)
        if self.classes:
            index = {v: i for i, v in enumerate(self.classes)}
            y = [index[v] for v in y]
        target = np.ascontiguousarray(y, dtype=np.float64)
        fit = (
            self._fit_polygrad
            if self.params.get("backend") == "polygrad"
            else lib.wl_sym_family_fit
        )
        handle = fit(
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
        self.artifact_version = 3 if self.params.get("hierarchical") else 2
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
            return self.predict_polygrad(X, kwargs.get("polygrad"))
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
            kind="sym.family.formula@3"
            if self.artifact_version == 3
            else "sym.pg-family.formula@1",
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
        if self.artifact_version == 3:
            out.update(
                nFeatures=self.n_features, sourceEncoding="inputs-then-earlier-terms"
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
            obj.artifact_version = int.from_bytes(data[12:16], "little")
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

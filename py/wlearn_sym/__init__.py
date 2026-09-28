"""wlearn_sym -- symbolic models backed by the wlearn C11 core."""

import ctypes
import json

import numpy as np

from wlearn.bundle import decode_bundle, encode_bundle, write_bundle_output
from wlearn.registry import register

from ._family_c import CFamilyEngine
from ._strategy import resolve_strategy, family_search_space
from ._ffi import _DP, _I, get_lib, last_error
from ._family_format import PG_FAMILY_MEDIA

FAMILY_MEDIA = "application/vnd.wlearn.sym.family"
from ._polygrad import evaluate_formula_polygrad, refine_formula_polygrad
from ._verifier import FormulaVerifier

TYPE_ID_REGRESSOR = "wlearn.sym.regressor@1"
TYPE_ID_CLASSIFIER = "wlearn.sym.classifier@1"
TYPE_ID_TRANSFORMER = "wlearn.sym.transformer@1"

TASK = {"regression": 0, "classification": 1, "transformer": 2}
LOSS = {"mse": 0, "mae": 1, "huber": 2, "logloss": 3}
OPSET = {"basic": 0, "smooth": 1, "full": 2}
FINAL_SELECTOR = {
    "objective": 0,
    "loss": 1,
    "validloss": 1,
    "validationloss": 1,
    "score": 2,
}
ENGINE_C = "c"
ENGINE_PG_FAMILY = "pg-family"


def _resolve(mapping, value, fallback):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return mapping.get(value.lower(), fallback)
    return fallback


def _as_matrix(X, n_features=None):
    arr = np.ascontiguousarray(X, dtype=np.float64)
    if arr.ndim == 1:
        if n_features:
            if arr.size % n_features != 0:
                raise ValueError(
                    f"flat X length {arr.size} is not divisible by n_features {n_features}"
                )
            arr = arr.reshape(-1, n_features)
        else:
            arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {arr.ndim}")
    if not np.isfinite(arr).all():
        raise ValueError("X must contain only finite values")
    return arr


def _as_y(y, rows):
    arr = np.ascontiguousarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got {arr.ndim}")
    if arr.shape[0] != rows:
        raise ValueError(f"y length ({arr.shape[0]}) does not match X rows ({rows})")
    if not np.isfinite(arr).all():
        raise ValueError("y must contain only finite values")
    return arr


def _mse(y, pred):
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.mean(err * err))


def _unique_sorted(y):
    return sorted(float(v) for v in np.unique(y))


def _map_classes(y, classes):
    index = {float(c): i for i, c in enumerate(classes)}
    out = np.zeros(y.shape[0], dtype=np.float64)
    for i, value in enumerate(y):
        key = float(value)
        if key not in index:
            raise ValueError(f"unknown class label: {key}")
        out[i] = index[key]
    return out


class _BaseSym:
    type_id = None
    task_name = None

    def __init__(self, params=None):
        self._params = dict(params or {})
        self._handle = None
        self._fitted = False
        self._disposed = False
        self._n_features = 0
        self._n_outputs = 0
        self._n_classes = 0
        self._classes = []
        self._choice = resolve_strategy(self._params, self.task_name)
        self._params.update(strategy=self._choice[0], backend=self._choice[1])
        self._engine_name = (
            ENGINE_PG_FAMILY if self._choice[1] == "polygrad" else ENGINE_C
        )
        self._family_engine = None

    @classmethod
    def create(cls, params=None):
        return cls(params)

    def fit(self, X, y):
        if self._disposed:
            raise RuntimeError(f"{self.__class__.__name__} has been disposed")
        X = _as_matrix(X)
        y = _as_y(y, X.shape[0])
        if self._choice[0] == "family":
            classes = (
                list(self._params.get("classes", _unique_sorted(y)))
                if self.task_name == "classification"
                else []
            )
            if self.task_name == "classification" and (
                len(classes) < 2
                or len(set(classes)) != len(classes)
                or not np.isfinite(classes).all()
            ):
                raise ValueError("classification requires unique finite classes")
            engine = CFamilyEngine(self._params, classes)
            try:
                engine.fit(X, y)
            except Exception:
                engine.dispose()
                raise
            self._free_handle()
            self._family_engine = engine
            self._classes = classes
            self._n_classes = len(classes)
            self._n_outputs = len(classes) if len(classes) > 2 else 1
            self._n_features = X.shape[1]
            self._fitted = True
            return self
        self._free_handle()
        lib = get_lib()
        fit_y = y
        n_classes = 0
        if self.task_name == "classification":
            self._classes = list(self._params.get("classes") or _unique_sorted(y))
            if len(self._classes) < 2:
                raise ValueError("classification requires at least two classes")
            n_classes = len(self._classes)
            fit_y = _map_classes(y, self._classes)

        handle = lib.wl_sym_fit(
            X.ctypes.data_as(_DP),
            X.shape[0],
            X.shape[1],
            fit_y.ctypes.data_as(_DP),
            TASK[self.task_name],
            n_classes,
            int(self._params.get("topK", self._params.get("top_k", 8))),
            int(self._params.get("population", 256)),
            int(self._params.get("generations", 120)),
            int(self._params.get("maxNodes", self._params.get("max_nodes", 31))),
            int(self._params.get("maxDepth", self._params.get("max_depth", 6))),
            int(
                self._params.get("frontierSize", self._params.get("frontier_size", 16))
            ),
            int(
                self._params.get(
                    "tournamentSize", self._params.get("tournament_size", 4)
                )
            ),
            int(self._params.get("eliteCount", self._params.get("elite_count", 4))),
            _resolve(
                LOSS,
                self._params.get("loss"),
                LOSS["logloss"] if self.task_name == "classification" else LOSS["mse"],
            ),
            _resolve(
                OPSET,
                self._params.get("operatorSet", self._params.get("operator_set")),
                OPSET["full"],
            ),
            int(
                self._params.get(
                    "earlyStopRounds", self._params.get("early_stop_rounds", 30)
                )
            ),
            int(self._params.get("seed", 42)),
            float(
                self._params.get(
                    "validationFraction", self._params.get("validation_fraction", 0)
                )
            ),
            float(
                self._params.get(
                    "complexityPenalty", self._params.get("complexity_penalty", 0.001)
                )
            ),
            float(
                self._params.get(
                    "mutationRate", self._params.get("mutation_rate", 0.35)
                )
            ),
            float(
                self._params.get(
                    "crossoverRate", self._params.get("crossover_rate", 0.55)
                )
            ),
            float(
                self._params.get(
                    "constantRate", self._params.get("constant_rate", 0.20)
                )
            ),
            float(self._params.get("constMin", self._params.get("const_min", -4.0))),
            float(self._params.get("constMax", self._params.get("const_max", 4.0))),
            float(self._params.get("huberDelta", self._params.get("huber_delta", 1.0))),
            float(self._params.get("tol", 1e-12)),
            int(self._params.get("islands", 1)),
            int(
                self._params.get(
                    "migrationInterval", self._params.get("migration_interval", 0)
                )
            ),
            int(
                self._params.get(
                    "migrationCount", self._params.get("migration_count", 0)
                )
            ),
            int(
                self._params.get(
                    "warmupGenerations", self._params.get("warmup_generations", 0)
                )
            ),
            int(
                self._params.get(
                    "warmupMinNodes", self._params.get("warmup_min_nodes", 0)
                )
            ),
            int(self._params.get("broodSize", self._params.get("brood_size", 1))),
            int(
                self._params.get(
                    "rowSampleSize", self._params.get("row_sample_size", 0)
                )
            ),
            int(
                self._params.get(
                    "localRefineInterval", self._params.get("local_refine_interval", 0)
                )
            ),
            int(
                self._params.get(
                    "localRefineCount", self._params.get("local_refine_count", 0)
                )
            ),
            int(
                self._params.get(
                    "complexityHofSize", self._params.get("complexity_hof_size", 0)
                )
            ),
            _resolve(
                FINAL_SELECTOR,
                self._params.get("finalSelector", self._params.get("final_selector")),
                FINAL_SELECTOR["objective"],
            ),
            float(
                self._params.get(
                    "complexityBucketWidth",
                    self._params.get("complexity_bucket_width", 2.0),
                )
            ),
        )
        if not handle:
            raise RuntimeError(f"Symbolic fit failed: {last_error(lib)}")
        self._handle = handle
        self._fitted = True
        self._n_features = int(lib.wl_sym_get_n_features(handle))
        self._n_outputs = int(lib.wl_sym_get_n_outputs(handle))
        self._n_classes = int(lib.wl_sym_get_n_classes(handle))
        return self

    def predict(self, X):
        self._ensure_fitted()
        if self._family_engine is not None:
            return self._family_engine.predict(X)
        lib = get_lib()
        X = _as_matrix(X, self._n_features)
        out_len = (
            X.shape[0] * self._n_outputs
            if self.task_name == "transformer"
            else X.shape[0]
        )
        out = np.zeros(out_len, dtype=np.float64)
        ret = lib.wl_sym_predict(
            self._handle,
            X.ctypes.data_as(_DP),
            X.shape[0],
            X.shape[1],
            out.ctypes.data_as(_DP),
        )
        if ret != 0:
            raise RuntimeError(f"Symbolic predict failed: {last_error(lib)}")
        if self.task_name == "classification" and self._classes:
            mapped = np.asarray([self._classes[int(v)] for v in out], dtype=np.float64)
            return mapped
        if self.task_name == "transformer":
            return out.reshape(X.shape[0], self._n_outputs)
        return out

    def decision_function(self, X):
        self._ensure_fitted()
        if self._family_engine is not None:
            if hasattr(self._family_engine, "decision_function"):
                return self._family_engine.decision_function(X)
            return self._family_engine.predict(X)
        lib = get_lib()
        X = _as_matrix(X, self._n_features)
        out = np.zeros(X.shape[0] * self._n_outputs, dtype=np.float64)
        ret = lib.wl_sym_predict_raw(
            self._handle,
            X.ctypes.data_as(_DP),
            X.shape[0],
            X.shape[1],
            out.ctypes.data_as(_DP),
        )
        if ret != 0:
            raise RuntimeError(f"Symbolic decision_function failed: {last_error(lib)}")
        return out.reshape(X.shape[0], self._n_outputs)

    decisionFunction = decision_function

    def score(self, X, y):
        self._ensure_fitted()
        if self._family_engine is not None:
            return self._family_engine.score(X, y)
        lib = get_lib()
        X = _as_matrix(X, self._n_features)
        y = _as_y(y, X.shape[0])
        fit_y = (
            _map_classes(y, self._classes)
            if self.task_name == "classification" and self._classes
            else y
        )
        result = lib.wl_sym_score(
            self._handle,
            X.ctypes.data_as(_DP),
            X.shape[0],
            X.shape[1],
            fit_y.ctypes.data_as(_DP),
        )
        if not np.isfinite(result):
            raise RuntimeError(f"Symbolic score failed: {last_error(lib)}")
        return float(result)

    def formula(self, format="json", index=0):
        self._ensure_fitted()
        if self._family_engine is not None:
            return self._family_engine.formula(format=format, index=index)
        data = self._formula_bytes(index, format)
        text = data.decode("utf-8")
        return text if format == "text" else json.loads(text)

    def frontier(self):
        self._ensure_fitted()
        if self._family_engine is not None:
            return self._family_engine.frontier()
        lib = get_lib()
        n_formulas = int(lib.wl_sym_get_n_formulas(self._handle))
        n = int(lib.wl_sym_get_frontier_size(self._handle))
        out = []
        for i in range(n):
            idx = n_formulas + i
            item = self.formula(index=idx)
            item["index"] = i
            item["text"] = self.formula(format="text", index=idx)
            out.append(item)
        return out

    def verify(self, checks=None):
        return FormulaVerifier.verify(
            self.formula(), {"n_features": self._n_features, **(checks or {})}
        )

    def predict_polygrad(self, X, polygrad=None):
        self._ensure_fitted()
        if self._choice[0] == "family":
            return self._family_engine.predict_polygrad(X, polygrad)
        if self._family_engine is not None:
            return self._family_engine.predict(X)
        X = _as_matrix(X, self._n_features)
        formulas = [
            self.formula(index=i)
            for i in range(self._n_outputs if self.task_name == "transformer" else 1)
        ]
        cols = [
            evaluate_formula_polygrad(formula, X, polygrad=polygrad)
            for formula in formulas
        ]
        if len(cols) == 1:
            return cols[0]
        return np.column_stack(cols)

    predictPolygrad = predict_polygrad

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
        self._ensure_fitted()
        if isinstance(self._family_engine, CFamilyEngine):
            return self._family_engine.refine_polygrad(
                X,
                y,
                index=index,
                epochs=epochs,
                lr=lr,
                optimizer=optimizer,
                tolerance=tolerance,
                polygrad=polygrad,
            )
        if self.task_name != "regression":
            raise ValueError(
                "refine_polygrad currently supports regression formulas only"
            )
        if index != 0:
            raise ValueError(
                "Refinement supports the selected regression formula (index 0)"
            )
        X = _as_matrix(X, self._n_features)
        y = _as_y(y, X.shape[0])
        before_pred = self.predict(X)
        before_loss = _mse(y, before_pred)
        before_score = self.score(X, y)
        formula = self.formula(index=index)
        report = refine_formula_polygrad(
            formula, X, y, epochs=epochs, lr=lr, optimizer=optimizer, polygrad=polygrad
        )
        if report.get("status") != "ok":
            report.update(
                {
                    "committed": False,
                    "beforeLoss": before_loss,
                    "beforeScore": before_score,
                }
            )
            return report
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        # Validate with the authoritative C evaluator before changing this model.
        candidate = type(self).load(self.save())
        try:
            for item in report["committedConstants"]:
                candidate._set_formula_constant(
                    index, int(item["nodeIndex"]), float(item["value"])
                )
            after_loss = _mse(y, candidate.predict(X))
            accepted = bool(
                np.isfinite(after_loss) and after_loss <= before_loss + tolerance
            )
            if accepted:
                # Search-time metrics belong to the original split, not these rows.
                self._handle, candidate._handle = candidate._handle, self._handle
            report.update(
                dict(
                    loss=after_loss,
                    status=(
                        "improved"
                        if after_loss + tolerance < before_loss
                        else "unchanged"
                    )
                    if accepted
                    else "rejected",
                    committed=accepted,
                    beforeLoss=before_loss,
                    beforeScore=before_score,
                    afterLoss=after_loss,
                    afterScore=self.score(X, y),
                )
            )
            return report
        finally:
            candidate.dispose()

    refinePolygrad = refine_polygrad

    def save(self, path=None):
        self._ensure_fitted()
        if self._family_engine is not None:
            binary = getattr(self._family_engine, "artifact_version", 1) >= 2
            data = (
                self._family_engine.to_bytes()
                if binary
                else json.dumps(
                    self._family_engine.to_state(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            bundle = encode_bundle(
                {
                    "typeId": self.type_id.replace(
                        "@1", f"@{self._family_engine.artifact_version}"
                    )
                    if binary
                    else self.type_id,
                    "params": self.get_params(),
                    "metadata": {
                        "strategy": "family",
                        "backend": self._choice[1],
                        "nFeatures": self._n_features,
                        "nOutputs": self._n_outputs,
                        "nClasses": self._n_classes,
                        "classes": self._classes,
                        "task": self.task_name,
                    },
                },
                [
                    {
                        "id": "model",
                        "mediaType": FAMILY_MEDIA if binary else PG_FAMILY_MEDIA,
                        "data": data,
                    }
                ],
            )
            return write_bundle_output(bundle, path)
        raw = self._save_raw()
        bundle = encode_bundle(
            {
                "typeId": self.type_id,
                "params": self.get_params(),
                "metadata": {
                    "nFeatures": self._n_features,
                    "nOutputs": self._n_outputs,
                    "nClasses": self._n_classes,
                    "classes": self._classes,
                    "task": self.task_name,
                },
            },
            [
                {
                    "id": "model",
                    "mediaType": "application/vnd.wlearn.sym.raw",
                    "data": raw,
                }
            ],
        )
        return write_bundle_output(bundle, path)

    @classmethod
    def load(cls, data):
        manifest, toc, blobs = decode_bundle(data)
        return cls._from_bundle(manifest, toc, blobs)

    @classmethod
    def _from_bundle(cls, manifest, toc, blobs):
        binary = (
            manifest.get("typeId")
            in [cls.type_id.replace("@1", f"@{v}") for v in (2, 3)]
            and cls.task_name != "transformer"
        )
        if manifest.get("typeId") != cls.type_id and not binary:
            raise ValueError(f"Unsupported sym bundle typeId: {manifest.get('typeId')}")
        entry = next((item for item in toc if item["id"] == "model"), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')
        raw = bytes(blobs[entry["offset"] : entry["offset"] + entry["length"]])
        metadata = manifest.get("metadata") or {}
        if binary:
            params = manifest.get("params") or {}
            if (
                entry.get("mediaType") != FAMILY_MEDIA
                or params.get("strategy") != "family"
            ):
                raise ValueError("invalid family artifact type or strategy")
            obj = cls(params)
            try:
                classes = metadata.get("classes")
                if not isinstance(classes, list) or (
                    cls.task_name == "classification"
                ) != bool(classes):
                    raise ValueError("family payload task mismatch")
                obj._family_engine = CFamilyEngine.from_bytes(raw, obj._params, classes)
                if not manifest["typeId"].endswith(
                    f"@{obj._family_engine.artifact_version}"
                ):
                    raise ValueError("family artifact semantics/typeId mismatch")
                obj._n_features = obj._family_engine.n_features
                obj._classes = list(classes)
                obj._n_classes = len(classes)
                obj._n_outputs = len(classes) if len(classes) > 2 else 1
                if (
                    metadata.get("nFeatures") != obj._n_features
                    or metadata.get("nClasses") != obj._n_classes
                    or metadata.get("nOutputs") != obj._n_outputs
                ):
                    raise ValueError("family artifact dimensions mismatch")
                obj._fitted = True
                return obj
            except Exception:
                obj.dispose()
                raise
        if entry.get("mediaType") == FAMILY_MEDIA:
            raise ValueError("family binary requires an @2 or @3 typeId")
        if (
            entry.get("mediaType") == PG_FAMILY_MEDIA
            or metadata.get("engine") == ENGINE_PG_FAMILY
        ):
            if cls.task_name == "transformer":
                raise ValueError(
                    "pg-family bundles are supported for SymbolicRegressor and SymbolicClassifier only"
                )
            state = json.loads(raw.decode("utf-8"))
            params = manifest.get("params") or {}
            expected = (
                "classifier" if cls.task_name == "classification" else "regressor"
            )
            if (
                not isinstance(state, dict)
                or state.get("kind") != f"wlearn.sym.pg-family.{expected}@1"
            ):
                raise ValueError("family payload task does not match bundle type")
            params = {
                **params,
                "strategy": "family",
                "backend": params.get("backend", "polygrad"),
            }
            params.pop("engine", None)
            obj = cls(params)
            obj._family_engine = CFamilyEngine.from_state(state, params)
            obj._fitted = True
            obj._n_features = obj._family_engine.n_features
            obj._classes = obj._family_engine.classes
            obj._n_classes = len(obj._classes)
            obj._n_outputs = obj._n_classes if obj._n_classes > 2 else 1
            return obj
        return cls._load_raw(
            raw, manifest.get("params") or {}, manifest.get("metadata") or {}
        )

    @classmethod
    def _load_raw(cls, data, params=None, metadata=None):
        lib = get_lib()
        buf = ctypes.create_string_buffer(bytes(data))
        handle = lib.wl_sym_load(buf, len(data))
        if not handle:
            raise RuntimeError(f"Symbolic load failed: {last_error(lib)}")
        obj = cls(params)
        obj._handle = handle
        obj._fitted = True
        obj._n_features = int(lib.wl_sym_get_n_features(handle))
        obj._n_outputs = int(lib.wl_sym_get_n_outputs(handle))
        obj._n_classes = int(lib.wl_sym_get_n_classes(handle))
        obj._classes = list((metadata or {}).get("classes") or [])
        return obj

    def _save_raw(self):
        lib = get_lib()
        out_buf = ctypes.c_void_p()
        out_len = _I()
        ret = lib.wl_sym_save(
            self._handle, ctypes.byref(out_buf), ctypes.byref(out_len)
        )
        if ret != 0:
            raise RuntimeError(f"Symbolic save failed: {last_error(lib)}")
        try:
            return bytes((ctypes.c_ubyte * out_len.value).from_address(out_buf.value))
        finally:
            lib.wl_sym_free_buffer(out_buf)

    def _formula_bytes(self, index, format):
        lib = get_lib()
        out_buf = ctypes.c_void_p()
        out_len = _I()
        fn = lib.wl_sym_formula_text if format == "text" else lib.wl_sym_formula_json
        ret = fn(self._handle, int(index), ctypes.byref(out_buf), ctypes.byref(out_len))
        if ret != 0:
            raise RuntimeError(f"Formula export failed: {last_error(lib)}")
        try:
            return bytes((ctypes.c_ubyte * out_len.value).from_address(out_buf.value))
        finally:
            lib.wl_sym_free_buffer(out_buf)

    def _set_formula_constant(self, index, node_index, value):
        lib = get_lib()
        ret = lib.wl_sym_set_formula_constant(
            self._handle, int(index), int(node_index), float(value)
        )
        if ret != 0:
            raise RuntimeError(f"Symbolic constant update failed: {last_error(lib)}")

    def _set_formula_metrics(self, index, train_loss, valid_loss, objective):
        lib = get_lib()
        ret = lib.wl_sym_set_formula_metrics(
            self._handle,
            int(index),
            float(train_loss),
            float(valid_loss),
            float(objective),
        )
        if ret != 0:
            raise RuntimeError(f"Symbolic metric update failed: {last_error(lib)}")

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._free_handle()

    def get_params(self):
        params = {
            k: v
            for k, v in self._params.items()
            if k not in {"polygrad", "polygradRuntime", "wasm", "_opIds"}
        }
        return {**params, "strategy": self._choice[0], "backend": self._choice[1]}

    def set_params(self, params=None, **kwargs):
        merged = {**self._params, **(params or {}), **kwargs}
        choice = resolve_strategy(merged, self.task_name)
        if choice != self._choice:
            self._free_handle()
        self._params, self._choice = merged, choice
        self._engine_name = ENGINE_PG_FAMILY if choice[1] == "polygrad" else ENGINE_C
        return self

    @property
    def capabilities(self):
        family = self._choice[0] == "family"
        return {
            "classifier": self.task_name == "classification",
            "regressor": self.task_name == "regression",
            "transformer": self.task_name == "transformer",
            "predictProba": self.task_name == "classification",
            "sampleWeight": False,
            "csr": False,
            "strategies": ["tree"]
            if self.task_name == "transformer"
            else ["tree", "family"],
            "backends": ["c", "polygrad"] if family else ["c"],
            "activeStrategy": self._choice[0],
            "activeBackend": self._choice[1],
            "polygradRefinement": (family or self.task_name == "regression"),
        }

    getParams = get_params
    setParams = set_params

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_outputs(self):
        return self._n_outputs

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def classes(self):
        return list(self._classes)

    def _ensure_fitted(self):
        if self._disposed:
            raise RuntimeError(f"{self.__class__.__name__} has been disposed")
        if not self._fitted or (not self._handle and self._family_engine is None):
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first"
            )

    def _free_handle(self):
        if self._handle:
            get_lib().wl_sym_free(self._handle)
            self._handle = None
        if self._family_engine is not None:
            self._family_engine.dispose()
            self._family_engine = None
        self._fitted = False

    def __del__(self):
        if (
            getattr(self, "_handle", None) or getattr(self, "_family_engine", None)
        ) and not getattr(self, "_disposed", True):
            try:
                self.dispose()
            except Exception:
                pass


class SymbolicRegressor(_BaseSym):
    type_id = TYPE_ID_REGRESSOR
    task_name = "regression"

    @staticmethod
    def default_search_space():
        return family_search_space(
            {
                "population": {"type": "int_uniform", "low": 128, "high": 1024},
                "generations": {"type": "int_uniform", "low": 80, "high": 400},
                "maxNodes": {"type": "int_uniform", "low": 7, "high": 63},
                "operatorSet": {
                    "type": "categorical",
                    "values": ["full", "smooth", "basic"],
                },
                "complexityPenalty": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
                "mutationRate": {"type": "uniform", "low": 0.15, "high": 0.55},
                "crossoverRate": {"type": "uniform", "low": 0.35, "high": 0.75},
                "islands": {"type": "categorical", "values": [1, 2, 4]},
                "broodSize": {"type": "categorical", "values": [1, 2]},
                "complexityHofSize": {"type": "categorical", "values": [0, 64, 128]},
                "finalSelector": {
                    "type": "categorical",
                    "values": ["objective", "loss"],
                },
                "loss": {"type": "categorical", "values": ["mse", "mae", "huber"]},
            }
        )

    defaultSearchSpace = default_search_space


class SymbolicClassifier(_BaseSym):
    type_id = TYPE_ID_CLASSIFIER
    task_name = "classification"

    def predict_proba(self, X):
        self._ensure_fitted()
        if self._family_engine is not None:
            return self._family_engine.predict_proba(X)
        lib = get_lib()
        X = _as_matrix(X, self._n_features)
        n_classes = self._n_classes or len(self._classes) or 2
        out = np.zeros(X.shape[0] * n_classes, dtype=np.float64)
        ret = lib.wl_sym_predict_proba(
            self._handle,
            X.ctypes.data_as(_DP),
            X.shape[0],
            X.shape[1],
            out.ctypes.data_as(_DP),
        )
        if ret != 0:
            raise RuntimeError(f"Symbolic predict_proba failed: {last_error(lib)}")
        return out.reshape(X.shape[0], n_classes)

    predictProba = predict_proba

    @staticmethod
    def default_search_space():
        return family_search_space(
            {
                "population": {"type": "int_uniform", "low": 128, "high": 1024},
                "generations": {"type": "int_uniform", "low": 80, "high": 400},
                "maxNodes": {"type": "int_uniform", "low": 7, "high": 63},
                "operatorSet": {
                    "type": "categorical",
                    "values": ["full", "smooth", "basic"],
                },
                "complexityPenalty": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
                "mutationRate": {"type": "uniform", "low": 0.15, "high": 0.55},
                "crossoverRate": {"type": "uniform", "low": 0.35, "high": 0.75},
                "islands": {"type": "categorical", "values": [1, 2, 4]},
                "broodSize": {"type": "categorical", "values": [1, 2]},
                "complexityHofSize": {"type": "categorical", "values": [0, 64, 128]},
                "finalSelector": {
                    "type": "categorical",
                    "values": ["objective", "loss"],
                },
            }
        )

    defaultSearchSpace = default_search_space


class FormulaTransformer(_BaseSym):
    type_id = TYPE_ID_TRANSFORMER
    task_name = "transformer"

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    fitTransform = fit_transform

    @staticmethod
    def default_search_space():
        return {
            "topK": {"type": "int_uniform", "low": 2, "high": 16},
            "population": {"type": "int_uniform", "low": 128, "high": 1024},
            "generations": {"type": "int_uniform", "low": 80, "high": 400},
            "maxNodes": {"type": "int_uniform", "low": 7, "high": 63},
            "operatorSet": {
                "type": "categorical",
                "values": ["full", "smooth", "basic"],
            },
            "complexityPenalty": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
            "islands": {"type": "categorical", "values": [1, 2, 4]},
            "broodSize": {"type": "categorical", "values": [1, 2]},
            "complexityHofSize": {"type": "categorical", "values": [0, 64, 128]},
            "finalSelector": {"type": "categorical", "values": ["objective", "loss"]},
        }

    defaultSearchSpace = default_search_space


def register_sym_loaders():
    register(TYPE_ID_REGRESSOR.replace("@1", "@3"), SymbolicRegressor._from_bundle)
    register(TYPE_ID_CLASSIFIER.replace("@1", "@3"), SymbolicClassifier._from_bundle)
    register(TYPE_ID_REGRESSOR.replace("@1", "@2"), SymbolicRegressor._from_bundle)
    register(TYPE_ID_CLASSIFIER.replace("@1", "@2"), SymbolicClassifier._from_bundle)
    register(TYPE_ID_REGRESSOR, SymbolicRegressor._from_bundle)
    register(TYPE_ID_CLASSIFIER, SymbolicClassifier._from_bundle)
    register(TYPE_ID_TRANSFORMER, FormulaTransformer._from_bundle)


registerSymLoaders = register_sym_loaders
register_sym_loaders()

__all__ = [
    "SymbolicRegressor",
    "SymbolicClassifier",
    "FormulaTransformer",
    "FormulaVerifier",
    "register_sym_loaders",
    "registerSymLoaders",
]

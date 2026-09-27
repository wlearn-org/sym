import numpy as np


def _resolve_polygrad_runtime(polygrad=None):
    if polygrad is not None and hasattr(polygrad, "Tensor"):
        return polygrad, False
    import polygrad as pg_module

    if isinstance(polygrad, dict):
        device = polygrad.get("device", "auto")
        return pg_module.create(device=device), True
    return pg_module.create(device="cpu"), True


def _build_formula_tensor(Tensor, x, formula, rows, const_params=None):
    vals = []
    const_params = const_params or {}
    for i, node in enumerate(formula["nodes"]):
        op = node["op"]
        if op == "const":
            out = const_params.get(i)
            if out is None:
                out = Tensor.full((rows,), float(node["value"]), dtype="float32", buffer=False)
        elif op == "var":
            out = x[:, int(node["feature"])]
        elif op == "add":
            out = vals[node["left"]] + vals[node["right"]]
        elif op == "sub":
            out = vals[node["left"]] - vals[node["right"]]
        elif op == "mul":
            out = vals[node["left"]] * vals[node["right"]]
        elif op == "div":
            den_raw = vals[node["right"]]
            near_zero = den_raw.abs() < 1e-12
            signed_eps = (den_raw >= 0).where(1e-12, -1e-12)
            out = vals[node["left"]] / near_zero.where(signed_eps, den_raw)
        elif op == "neg":
            out = -vals[node["left"]]
        elif op == "abs":
            out = vals[node["left"]].abs()
        elif op == "sqrt":
            out = vals[node["left"]].abs().sqrt()
        elif op == "log":
            out = (vals[node["left"]].abs() + 1e-12).log()
        elif op == "exp":
            out = vals[node["left"]].maximum(-40).minimum(40).exp()
        elif op == "sin":
            out = vals[node["left"]].sin()
        elif op == "cos":
            out = vals[node["left"]].cos()
        elif op == "tanh":
            out = vals[node["left"]].tanh()
        elif op == "min":
            out = vals[node["left"]].minimum(vals[node["right"]])
        elif op == "max":
            out = vals[node["left"]].maximum(vals[node["right"]])
        else:
            raise ValueError(f"unsupported formula op for Polygrad: {op}")
        # Match the C evaluator after every node, in float32 arithmetic.
        vals.append(out.isfinite().where(out, 0).maximum(-1e12).minimum(1e12))
    return vals[-1]


def evaluate_formula_polygrad(formula, X, polygrad=None):
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    rows, cols = X.shape
    pg, owns_runtime = _resolve_polygrad_runtime(polygrad)
    x = model = None
    try:
        x = pg.Tensor.empty((rows, cols), dtype="float32")
        model = pg.Model.from_callable(
            lambda sym_x: _build_formula_tensor(pg.Tensor, sym_x, formula, rows),
            inputs={"sym_x": x})
        return np.asarray(model.forward(sym_x=X)["output"], dtype=np.float64)
    finally:
        if model is not None:
            model.dispose()
        if x is not None:
            x.dispose()
        if owns_runtime:
            pg.dispose()


_REFINABLE_OPS = {
    "const",
    "var",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs",
    "sqrt",
    "log",
    "sin",
    "cos",
    "tanh",
    "exp",
    "min",
    "max",
}


def _unsupported_refine_op(formula):
    for node in formula["nodes"]:
        if node["op"] not in _REFINABLE_OPS:
            return node["op"]
    return None


def _mse(y, pred):
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.mean(err * err))


def refine_formula_polygrad(formula, X, y, epochs=80, lr=0.001, optimizer="adam", polygrad=None):
    unsupported = _unsupported_refine_op(formula)
    if unsupported:
        return {
            "status": "unsupported",
            "reason": f"operator '{unsupported}' is not in the differentiable Polygrad refinement subset",
            "committedConstants": [],
        }

    const_nodes = [i for i, node in enumerate(formula["nodes"]) if node["op"] == "const"]
    if not const_nodes:
        return {
            "status": "unchanged",
            "reason": "formula has no constants to refine",
            "committedConstants": [],
        }

    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.ascontiguousarray(y, dtype=np.float32).reshape(-1)
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"y length ({y.shape[0]}) does not match X rows ({X.shape[0]})")

    rows, cols = X.shape
    pg, owns_runtime = _resolve_polygrad_runtime(polygrad)
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1 or not np.isfinite(lr) or lr <= 0:
        if owns_runtime:
            pg.dispose()
        raise ValueError("epochs must be positive integer and lr positive finite")
    x = target = model = None
    params = {}
    try:
        x = pg.Tensor.empty((rows, cols), dtype="float32")
        target = pg.Tensor.empty((rows,), dtype="float32")
        for i in const_nodes:
            params[f"c{i}"] = pg.Tensor([float(formula["nodes"][i]["value"])],
                                        dtype="float32")
        const_params = {i: params[f"c{i}"] for i in const_nodes}
        model = pg.Model.from_callable(
            lambda sym_refine_x: _build_formula_tensor(pg.Tensor, sym_refine_x, formula, rows, const_params),
            inputs={"sym_refine_x": x}, targets={"sym_refine_y": target},
            loss=lambda pred, sym_refine_y: (pred - sym_refine_y).square().mean(),
            params=params)
        history = model.fit({"sym_refine_x": X, "sym_refine_y": y},
                            epochs=epochs, optimizer=optimizer, lr=float(lr))
        refined = {**formula, "nodes": [dict(node) for node in formula["nodes"]]}
        committed = []
        for i in const_nodes:
            # Model owns the trained state; captured Tensor handles need not alias it.
            value = float(model.read_buffer(f"c{i}")[0])
            if not np.isfinite(value):
                raise RuntimeError("Polygrad refinement produced a non-finite constant")
            refined["nodes"][i]["value"] = value
            committed.append({"nodeIndex": i, "value": value})
        pred_after = evaluate_formula_polygrad(refined, X, polygrad=pg)
        return dict(status="ok", loss=_mse(y, pred_after), history=[float(v) for v in history],
                    formula=refined, committedConstants=committed)
    finally:
        if model is not None:
            model.dispose()
        for tensor in [x, target, *params.values()]:
            if tensor is not None:
                tensor.dispose()
        if owns_runtime:
            pg.dispose()

"""Public Polygrad Model graph for fixed-structure family parameter refinement."""

import numpy as np
from ._polygrad import _resolve_polygrad_runtime


def family_graph(Tensor, X, candidate, rows, cols, params, clip_parameters=True):
    values = []
    result = params["bias"]
    for t, op in enumerate(candidate["op"]):

        def source(index):
            return X[:, index] if index < cols else values[index - cols]

        a, b = source(candidate["fa"][t]), source(candidate["fb"][t])
        if op < 4:
            if op == 0:
                value = a + b
            elif op == 1:
                value = a - b
            elif op == 2:
                value = a * b
            else:
                den = (b.abs() < 1e-6).where((b < 0).where(-1e-6, 1e-6), b)
                value = a / den
        else:
            p0, p1 = params[f"p0_{t}"], params[f"p1_{t}"]
            if clip_parameters:
                p0, p1 = p0.maximum(-16).minimum(16), p1.maximum(-16).minimum(16)
            z = a * p0 + p1
            if op == 4:
                value = z.sin()
            elif op == 5:
                value = z.cos()
            elif op == 6:
                value = z.tanh()
            elif op == 7:
                value = (z.abs() + 1e-6).log()
            elif op == 8:
                value = z.abs().sqrt()
            else:
                value = z.maximum(-6).minimum(6).exp()
        values.append(value)
        result = result + params[f"coef_{t}"] * value
    return result.expand(rows)


def refine(candidate, X, y, *, epochs, lr, optimizer, ridge, polygrad):
    if type(epochs) is not int or epochs < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("epochs must be positive integer and lr positive finite")
    pg, owned = _resolve_polygrad_runtime(polygrad)
    model = x = target = None
    params = {}
    try:
        rows, cols = X.shape
        for t, op in enumerate(candidate["op"]):
            for key in (["p0", "p1"] if op >= 4 else []) + ["coef"]:
                params[f"{key}_{t}"] = pg.Tensor([candidate[key][t]], dtype="float32")
        params["bias"] = pg.Tensor([candidate["bias"]], dtype="float32")
        x = pg.Tensor.empty((rows, cols), dtype="float32")
        target = pg.Tensor.empty((rows,), dtype="float32")

        def loss(pred, family_y):
            penalty = sum(
                params[f"coef_{t}"].square().sum() for t in range(candidate["terms"])
            )
            return (pred - family_y).square().mean() + ridge * penalty

        model = pg.Model.from_callable(
            lambda family_x: family_graph(
                pg.Tensor, family_x, candidate, rows, cols, params
            ),
            inputs={"family_x": x},
            targets={"family_y": target},
            params=params,
            loss=loss,
        )
        history = model.fit(
            {
                "family_x": np.ascontiguousarray(X, dtype=np.float32),
                "family_y": np.ascontiguousarray(y, dtype=np.float32),
            },
            epochs=epochs,
            lr=lr,
            optimizer=optimizer,
        )
        updates = []
        for t, op in enumerate(candidate["op"]):
            updates.extend(
                [
                    float(model.read_buffer(f"p0_{t}")[0]) if op >= 4 else 1.0,
                    float(model.read_buffer(f"p1_{t}")[0]) if op >= 4 else 0.0,
                    float(model.read_buffer(f"coef_{t}")[0]),
                ]
            )
        updates.append(float(model.read_buffer("bias")[0]))
        return np.asarray(updates, dtype=np.float64), [float(v) for v in history]
    finally:
        if model is not None:
            model.dispose()
        for tensor in [x, target, *params.values()]:
            if tensor is not None:
                tensor.dispose()
        if owned:
            pg.dispose()


def predict(candidate, X, polygrad=None):
    from ._family_scorer import check_phases

    X = np.ascontiguousarray(X, dtype=np.float32).astype(np.float64)
    if not len(X):
        return np.empty(0)
    descriptors = np.array(
        [
            [
                list(v)
                for v in zip(*(candidate[k] for k in ("fa", "fb", "op", "p0", "p1")))
            ]
        ]
    )
    check_phases(np.max(np.abs(X), axis=0), descriptors, "float64")
    pg, owned = _resolve_polygrad_runtime(polygrad)
    x = model = None
    params = {}
    try:
        rows, cols = X.shape
        x = pg.Tensor.empty((rows, cols), dtype="float64")
        for t, op in enumerate(candidate["op"]):
            for key in (["p0", "p1"] if op >= 4 else []) + ["coef"]:
                params[f"{key}_{t}"] = pg.Tensor([candidate[key][t]], dtype="float64")
        params["bias"] = pg.Tensor([candidate["bias"]], dtype="float64")
        model = pg.Model.from_callable(
            lambda family_x: family_graph(
                pg.Tensor, family_x, candidate, rows, cols, params, False
            ),
            inputs={"family_x": x},
            params=params,
        )
        result = np.asarray(model.forward(family_x=X)["output"], dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite Polygrad family prediction")
        return result
    finally:
        if model is not None:
            model.dispose()
        for tensor in [x, *params.values()]:
            if tensor is not None:
                tensor.dispose()
        if owned:
            pg.dispose()

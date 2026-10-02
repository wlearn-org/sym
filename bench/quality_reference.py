"""Independent NumPy evaluation of exported regression formulas, not formula text."""

import numpy as np


def operation(op, a, b, epsilon, exp_bound):
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return a / np.where(abs(b) < epsilon, np.where(b < 0, -epsilon, epsilon), b)
    if op == "neg":
        return -a
    if op == "abs":
        return abs(a)
    if op in ("sqrt", "sqrtabs"):
        return np.sqrt(abs(a))
    if op in ("log", "logabs"):
        return np.log(abs(a) + epsilon)
    if op in ("exp", "expclamp"):
        return np.exp(np.clip(a, -exp_bound, exp_bound))
    if op == "sin":
        return np.sin(a)
    if op == "cos":
        return np.cos(a)
    if op == "tanh":
        return np.tanh(a)
    if op == "min":
        return np.minimum(a, b)
    if op == "max":
        return np.maximum(a, b)
    raise ValueError(f"unsupported exported operator {op}")


def predict(formula, X):
    with np.errstate(all="ignore"):
        if "nodes" in formula:
            values = []
            for node in formula["nodes"]:
                op = node["op"]
                if op == "var":
                    value = X[:, node["feature"]]
                elif op == "const":
                    value = np.full(len(X), node["value"])
                else:
                    a = values[node["left"]] if node["left"] >= 0 else 0
                    b = values[node["right"]] if node["right"] >= 0 else 0
                    value = operation(op, a, b, 1e-12, 40)
                values.append(
                    np.clip(np.where(np.isfinite(value), value, 0), -1e12, 1e12)
                )
            return values[-1]
        values = list(X.astype(np.float32).astype(float).T)
        out = np.full(len(X), formula["bias"], dtype=float)
        for term in formula["terms"]:
            a, b = values[term["featureA"]], values[term["featureB"]]
            if term["op"] not in ("add", "sub", "mul", "div"):
                a = a * term["p0"] + term["p1"]
            value = operation(term["op"], a, b, 1e-6, 6)
            values.append(value)
            out += term["coefficient"] * value
        return out.astype(np.float32).astype(float)

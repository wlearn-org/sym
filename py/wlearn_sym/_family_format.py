"""Shared existing family payload and display helpers; no search or runtime."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import numpy as np

OPS = ["add", "sub", "mul", "div", "sin", "cos", "tanh", "logabs", "sqrtabs", "expclamp"]
PG_FAMILY_MEDIA = "application/vnd.wlearn.sym.pg-family+json"
OP_PRESETS = {
    "basic": ["add", "sub", "mul", "div"],
    "smooth": ["add", "sub", "mul", "div", "sin", "cos", "tanh"],
    "full": OPS,
}
OP_INDEX = {name: i for i, name in enumerate(OPS)}


def resolve_operator_ids(params=None):
    params = params or {}
    requested = params.get("operators", params.get("operatorSet", params.get("operator_set", "full")))
    names = requested if isinstance(requested, (list, tuple)) else OP_PRESETS.get(str(requested).lower())
    if names is None:
        raise ValueError(f"unsupported pg-family operatorSet: {requested}")
    ids = []
    for value in names:
        op_id = int(value) if isinstance(value, (int, np.integer)) else OP_INDEX.get(str(value).lower())
        if op_id is None or op_id < 0 or op_id >= len(OPS):
            raise ValueError(f"unsupported pg-family operator: {value}")
        if op_id not in ids:
            ids.append(op_id)
    if not ids:
        raise ValueError("pg-family operators must not be empty")
    return ids


@dataclass
class Candidate:
    terms: int
    fa: np.ndarray
    fb: np.ndarray
    op: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    coef: np.ndarray
    bias: float
    loss: float = math.inf
    objective: float = math.inf
    complexity: float = 0.0
    generation: int = 0


def _candidate_to_plain(c: Candidate):
    return {
        "terms": int(c.terms),
        "fa": [int(v) for v in c.fa.tolist()],
        "fb": [int(v) for v in c.fb.tolist()],
        "op": [int(v) for v in c.op.tolist()],
        "p0": [float(v) for v in c.p0.tolist()],
        "p1": [float(v) for v in c.p1.tolist()],
        "coef": [float(v) for v in c.coef.tolist()],
        "bias": float(c.bias),
        "loss": float(c.loss),
        "objective": float(c.objective),
        "complexity": float(c.complexity),
        "generation": int(c.generation),
    }


def _candidate_from_plain(o: dict[str, Any]) -> Candidate:
    return Candidate(
        terms=int(o["terms"]),
        fa=np.asarray(o["fa"], dtype=np.int32),
        fb=np.asarray(o["fb"], dtype=np.int32),
        op=np.asarray(o["op"], dtype=np.int32),
        p0=np.asarray(o["p0"], dtype=np.float32),
        p1=np.asarray(o["p1"], dtype=np.float32),
        coef=np.asarray(o["coef"], dtype=np.float32),
        bias=float(o.get("bias", 0.0)),
        loss=float(o.get("loss", math.inf)),
        objective=float(o.get("objective", math.inf)),
        complexity=float(o.get("complexity", o.get("terms", 0) + 1)),
        generation=int(o.get("generation", 0)),
    )


def _formula_text(c: Candidate, feature_names):
    parts = []
    for t in range(c.terms):
        a = feature_names[int(c.fa[t])] if int(c.fa[t]) < len(feature_names) else f"x{int(c.fa[t])}"
        b = feature_names[int(c.fb[t])] if int(c.fb[t]) < len(feature_names) else f"x{int(c.fb[t])}"
        z = f"{float(c.p0[t]):.5g}*{a}+{float(c.p1[t]):.5g}"
        op = int(c.op[t])
        if op == 0:
            expr = f"({a}+{b})"
        elif op == 1:
            expr = f"({a}-{b})"
        elif op == 2:
            expr = f"({a}*{b})"
        elif op == 3:
            expr = f"protected_div({a},{b})"
        elif op == 4:
            expr = f"sin({z})"
        elif op == 5:
            expr = f"cos({z})"
        elif op == 6:
            expr = f"tanh({z})"
        elif op == 7:
            expr = f"log(abs({z})+1e-6)"
        elif op == 8:
            expr = f"sqrt(abs({z}))"
        else:
            expr = f"exp(clamp({z},-6,6))"
        parts.append(f"{float(c.coef[t]):.5g}*{expr}")
    return f"{float(c.bias):.5g} + " + " + ".join(parts)

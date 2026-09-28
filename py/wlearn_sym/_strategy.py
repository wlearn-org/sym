"""Resolve public strategy/backend choices without changing algorithms implicitly."""


def resolve_strategy(params, task):
    legacy = None
    if params.get("engine") is not None:
        engine = str(params["engine"]).lower()
        if engine == "pg-family":
            legacy = ("family", "polygrad")
        elif engine in ("c", "wasm", "c-wasm", "auto"):
            legacy = ("tree", "c")
        else:
            raise ValueError(f"unsupported sym engine: {engine}")
    strategy = params.get("strategy", legacy[0] if legacy else "tree")
    backend = params.get("backend", legacy[1] if legacy else "c")
    if strategy not in ("tree", "family") or backend not in ("c", "polygrad"):
        raise ValueError("strategy must be tree|family and backend must be c|polygrad")
    if legacy and (strategy, backend) != legacy[:2]:
        raise ValueError("engine conflicts with strategy/backend")
    if task == "transformer" and strategy != "tree":
        raise ValueError("family strategy does not support FormulaTransformer")
    if backend == "polygrad" and strategy == "tree":
        raise ValueError('tree search currently supports backend="c" only')
    return strategy, backend


def family_search_space(tree):
    space = {
        "strategy": {"type": "categorical", "values": ["tree", "family"]},
        "backend": {"type": "categorical", "values": ["c"]},
        **tree,
    }
    for key in ("maxNodes", "broodSize", "complexityHofSize", "finalSelector", "loss"):
        if key in space:
            space[key] = {**space[key], "condition": {"strategy": "tree"}}
    space["terms"] = {
        "type": "int_uniform",
        "low": 2,
        "high": 12,
        "condition": {"strategy": "family"},
    }
    space["ridge"] = {
        "type": "log_uniform",
        "low": 1e-8,
        "high": 1e-2,
        "condition": {"strategy": "family"},
    }
    space["immigrantRate"] = {
        "type": "categorical",
        "values": [0, 0.1, 0.2, 0.3],
        "condition": {"strategy": "family"},
    }
    return space

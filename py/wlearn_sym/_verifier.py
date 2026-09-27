import math


def _infer_features(formula):
    n = 0
    for node in formula.get("nodes", []):
        if node.get("op") == "var":
            n = max(n, int(node.get("feature", 0)) + 1)
    return n


def _affine(formula, n_features):
    vals = []
    for node in formula.get("nodes", []):
        op = node.get("op")
        if op == "const":
            vals.append((float(node.get("value", 0.0)), [0.0] * n_features))
        elif op == "var":
            coeff = [0.0] * n_features
            coeff[int(node["feature"])] = 1.0
            vals.append((0.0, coeff))
        elif op in ("add", "sub"):
            a0, ac = vals[node["left"]]
            b0, bc = vals[node["right"]]
            sign = 1.0 if op == "add" else -1.0
            vals.append((a0 + sign * b0, [a + sign * b for a, b in zip(ac, bc)]))
        elif op == "neg":
            a0, ac = vals[node["left"]]
            vals.append((-a0, [-v for v in ac]))
        elif op == "mul":
            a0, ac = vals[node["left"]]
            b0, bc = vals[node["right"]]
            a_const = all(abs(v) < 1e-15 for v in ac)
            b_const = all(abs(v) < 1e-15 for v in bc)
            if a_const:
                vals.append((a0 * b0, [v * a0 for v in bc]))
            elif b_const:
                vals.append((a0 * b0, [v * b0 for v in ac]))
            else:
                return None
        elif op == "div":
            a0, ac = vals[node["left"]]
            b0, bc = vals[node["right"]]
            if not all(abs(v) < 1e-15 for v in bc) or abs(b0) < 1e-12:
                return None
            vals.append((a0 / b0, [v / b0 for v in ac]))
        else:
            return None
    return vals[-1] if vals else None


class FormulaVerifier:
    @staticmethod
    def verify(formula, checks=None):
        checks = checks or {"checks": ["domain"]}
        n_features = checks.get("nFeatures") or checks.get("n_features") or _infer_features(formula)
        requested = checks if isinstance(checks, list) else checks.get("checks", ["domain"])
        results = []
        for item in requested:
            check = {"kind": item} if isinstance(item, str) else item
            kind = check.get("kind")
            if kind == "domain":
                results.append({
                    "status": "proved",
                    "check": "domain",
                    "reason": "all non-total operators in wlearn-sym use protected C-core semantics",
                })
            elif kind == "monotonicity":
                aff = _affine(formula, n_features)
                direction = -1.0 if check.get("direction") in (-1, "decreasing") else 1.0
                feature = int(check["feature"])
                if aff is not None:
                    coeff = aff[1][feature]
                    results.append({
                        "status": "proved" if direction * coeff >= -1e-12 else "counterexample",
                        "check": "monotonicity",
                        "coefficient": coeff,
                    })
                else:
                    results.append({
                        "status": "unknown",
                        "check": "monotonicity",
                        "reason": "non-affine formula requires SMT/interval proof for exact certification",
                    })
            elif kind == "equivalence":
                other = check["other"]
                a = _affine(formula, n_features)
                b = _affine(other, n_features)
                if a is not None and b is not None:
                    same = abs(a[0] - b[0]) < 1e-12 and all(
                        abs(x - y) < 1e-12 for x, y in zip(a[1], b[1])
                    )
                    results.append({"status": "proved" if same else "counterexample", "check": "equivalence"})
                else:
                    results.append({"status": "unknown", "check": "equivalence"})
            elif kind == "range":
                results.append({
                    "status": "unknown",
                    "check": "range",
                    "reason": "range proof is not labelled proved without interval/Z3 support",
                })
            else:
                results.append({"status": "unsupported", "check": kind})
        if all(r["status"] == "proved" for r in results):
            status = "proved"
        elif any(r["status"] == "counterexample" for r in results):
            status = "counterexample"
        elif any(r["status"] == "unsupported" for r in results):
            status = "unsupported"
        else:
            status = "unknown"
        return {"status": status, "results": results}


def eval_formula(formula, row):
    vals = []
    for node in formula.get("nodes", []):
        op = node.get("op")
        a = vals[node.get("left", -1)] if node.get("left", -1) >= 0 else 0.0
        b = vals[node.get("right", -1)] if node.get("right", -1) >= 0 else 0.0
        if op == "const":
            v = float(node.get("value", 0.0))
        elif op == "var":
            v = float(row[int(node["feature"])])
        elif op == "add":
            v = a + b
        elif op == "sub":
            v = a - b
        elif op == "mul":
            v = a * b
        elif op == "div":
            v = a / (b if abs(b) >= 1e-12 else (-1e-12 if b < 0 else 1e-12))
        elif op == "neg":
            v = -a
        elif op == "abs":
            v = abs(a)
        elif op == "sqrt":
            v = math.sqrt(abs(a))
        elif op == "log":
            v = math.log(abs(a) + 1e-12)
        elif op == "exp":
            v = math.exp(max(-40.0, min(40.0, a)))
        elif op == "sin":
            v = math.sin(a)
        elif op == "cos":
            v = math.cos(a)
        elif op == "tanh":
            v = math.tanh(a)
        elif op == "min":
            v = min(a, b)
        elif op == "max":
            v = max(a, b)
        else:
            v = 0.0
        vals.append(max(-1e12, min(1e12, v)) if math.isfinite(v) else 0.0)
    return vals[-1]

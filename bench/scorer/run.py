"""Matched fixed-candidate diagnostic; timings never presented as search fits."""

import argparse
import json
import time
import numpy as np
from resident import Reference, ResidentScorer


def data(rows, batch, terms, case, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (rows, 12)).astype(np.float32).astype(float)
    if case == "offset":
        X += 1e6
    if case == "collinear":
        X[:, 1] = X[:, 0] + 1e-4 * X[:, 1]
    if case == "constant":
        X[:, :3] = 1
    if case == "protected":
        X[:, 1] = rng.choice([0.0, 1e-8, -1e-8], rows)
    z = X - X.mean(0)
    y = np.ascontiguousarray(np.sin(3 * z[:, 0]) + z[:, 1] * z[:, 2] + 0.2 * z[:, 3])
    validation = np.ascontiguousarray((np.arange(rows) % 5 == 0).astype(np.uint8))
    d = np.empty((batch, terms, 5))
    d[:, :, :2] = rng.integers(0, X.shape[1], (batch, terms, 2))
    d[:, :, 2] = rng.integers(0, 10, (batch, terms))
    d[:, :, 3:] = rng.uniform(-2, 2, (batch, terms, 2)).astype(np.float32)
    # Current C canonicalization: arithmetic has no parameters; unary has no b.
    arithmetic = d[:, :, 2] < 4
    d[:, :, 3][arithmetic] = 1
    d[:, :, 4][arithmetic] = 0
    d[:, :, 1][~arithmetic] = 0
    return (
        X.astype(np.float32).astype(float),
        y.astype(np.float32).astype(float),
        validation,
        d,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--terms", type=int, default=6)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument(
        "--case",
        default="ordinary",
        choices=["ordinary", "offset", "collinear", "constant", "protected"],
    )
    p.add_argument("--mode", default="custom", choices=["custom", "tensor"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--device", default="CUDA")
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()
    X, y, validation, d = data(args.rows, args.batch, args.terms, args.case, args.seed)
    ref = Reference()
    scorer = ResidentScorer(
        X, y, validation, d, mode=args.mode, device=args.device, dtype=args.dtype
    )
    try:
        print(
            json.dumps(
                {
                    "event": "setup",
                    **vars(args),
                    "ms": scorer.setup_ms,
                    "schedules": [s.schedule_count for s in scorer.stages],
                    "runtime": scorer.rt.stats(),
                }
            ),
            flush=True,
        )
        rng = np.random.default_rng(args.seed + 1)
        for iteration in range(args.repeat):
            # Update descriptors on every replay; an unchanged cached output must fail.
            unary = d[:, :, 2] >= 4
            d[:, :, 3][unary] += rng.uniform(-0.03, 0.03, np.count_nonzero(unary))
            d[:, :, 3:] = d[:, :, 3:].astype(np.float32)
            before = time.perf_counter()
            coef, loss, phi = ref.fit(X, y, validation, d, 1e-8, features=False)
            c_ms = (time.perf_counter() - before) * 1000
            result = scorer.score(d, inspect=True)
            _, _, phi = ref.fit(X, y, validation, d, 1e-8, features=True)
            feature_error = np.abs(result["features"] - phi)
            pred = (
                np.einsum("bnt,bt->bn", phi, result["coefficients"][:, :-1])
                + result["coefficients"][:, -1:]
            )
            reference_pred = np.einsum("bnt,bt->bn", phi, coef[:, :-1]) + coef[:, -1:]
            actual_loss = np.stack(
                [
                    np.mean(
                        (pred[:, validation == 0] - y[validation == 0]) ** 2, axis=1
                    ),
                    np.mean(
                        (pred[:, validation != 0] - y[validation != 0]) ** 2, axis=1
                    ),
                ],
                axis=1,
            )
            print(
                json.dumps(
                    {
                        "event": "score",
                        "iteration": iteration,
                        "c_ms": c_ms,
                        "device_ms": result["ms"],
                        "max_feature_relative_error": float(
                            np.max(feature_error / (1 + np.abs(phi)))
                        ),
                        "max_prediction_rmse": float(
                            np.max(
                                np.sqrt(np.mean((pred - reference_pred) ** 2, axis=1))
                            )
                        ),
                        "max_device_loss_relative_error": float(
                            np.max(
                                np.abs(result["losses"] - actual_loss)
                                / (1 + actual_loss)
                            )
                        ),
                        "max_reference_loss_relative_error": float(
                            np.max(np.abs(result["losses"] - loss) / (1 + loss))
                        ),
                        "best_c": int(np.argmin(loss[:, 1])),
                        "best_device": int(np.argmin(result["losses"][:, 1])),
                        "traffic": result["traffic"],
                        "runtime": scorer.rt.stats(),
                    }
                ),
                flush=True,
            )
    finally:
        scorer.dispose()


if __name__ == "__main__":
    main()

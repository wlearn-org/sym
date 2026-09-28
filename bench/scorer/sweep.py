"""Score the SAME candidate pool using different device chunk sizes."""

import argparse
import json
import time

import numpy as np
from resident import Reference, ResidentScorer
from run import data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--pool", type=int, default=512)
    p.add_argument("--chunks", default="8,64,512")
    p.add_argument("--terms", type=int, default=6)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--mode", choices=["custom", "tensor"], default="custom")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="CUDA")
    a = p.parse_args()
    chunks = [int(v) for v in a.chunks.split(",")]
    if any(v < 1 or a.pool % v for v in chunks):
        p.error("chunks must divide the fixed pool")
    X, y, mask, initial = data(a.rows, a.pool, a.terms, "ordinary", 1729)
    ref = Reference()
    for chunk in chunks:
        d = initial.copy()
        scorer = ResidentScorer(
            X, y, mask, d[:chunk], mode=a.mode, dtype=a.dtype, device=a.device
        )
        try:
            first = time.perf_counter()
            scorer.score(d[:chunk])
            print(
                json.dumps(
                    {
                        "event": "setup",
                        "chunk": chunk,
                        "ms": scorer.setup_ms,
                        "first_score_ms": (time.perf_counter() - first) * 1000,
                        "settings": vars(a),
                    }
                ),
                flush=True,
            )
            for iteration in range(a.repeat):
                unary = d[:, :, 2] >= 4
                d[:, :, 3][unary] = (d[:, :, 3][unary] + 0.007).astype(np.float32)
                start = time.perf_counter()
                coef, expected, _ = ref.fit(X, y, mask, d, 1e-8)
                c_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                results = [
                    scorer.score(d[i : i + chunk]) for i in range(0, a.pool, chunk)
                ]
                losses = np.concatenate([r["losses"] for r in results])
                coefficients = np.concatenate([r["coefficients"] for r in results])
                device_ms = (time.perf_counter() - start) * 1000
                assert np.isfinite(coefficients).all()
                traffic = {
                    key: sum(r["traffic"][key] for r in results)
                    for key in results[0]["traffic"]
                }
                np.testing.assert_allclose(losses, expected, atol=1e-5, rtol=1e-5)
                print(
                    json.dumps(
                        {
                            "event": "pool",
                            "chunk": chunk,
                            "iteration": iteration,
                            "c_ms": c_ms,
                            "device_ms": device_ms,
                            "traffic": traffic,
                            "max_loss_difference": float(
                                np.max(np.abs(losses - expected))
                            ),
                            "best_c": int(np.argmin(expected[:, 1])),
                            "best_device": int(np.argmin(losses[:, 1])),
                            "device_owned_bytes": scorer.rt.stats()[
                                "buffer_owned_current_bytes"
                            ],
                        }
                    ),
                    flush=True,
                )
        finally:
            scorer.dispose()


if __name__ == "__main__":
    main()

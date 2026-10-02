"""Independent train/test data for the versioned quality screening manifest."""

import hashlib
import json
from pathlib import Path

import numpy as np


MANIFEST = Path(__file__).with_name("quality-tasks.json")


def manifest():
    return json.loads(MANIFEST.read_text())


def dataset(task, rows=256, test_rows=512):
    # Data are fixed per task. Search-seed comparisons see exactly the same split.
    identity = int.from_bytes(
        hashlib.sha256(task["id"].encode()).digest()[:4], "little"
    )
    seeds = np.random.SeedSequence([manifest()["data_seed"], identity]).spawn(2)
    rng, test_rng = [np.random.default_rng(s) for s in seeds]
    if task["function"] == "diabetes":
        from sklearn.datasets import load_diabetes

        X, y = load_diabetes(return_X_y=True)
        indices = rng.permutation(len(y))
        split = min(rows, int(0.7 * len(y)))
        train, test = indices[:split], indices[split:]
        return X[train], y[train], X[test], y[test]

    cols = task.get("columns", 5)
    train = rng.uniform(-1, 1, (rows, cols))
    test = test_rng.uniform(
        -task.get("test_range", 1), task.get("test_range", 1), (test_rows, cols)
    )

    def values(z):
        # Collinearity affects inputs and target consistently, before observations
        # are rounded to the same input precision seen by Sym and all baselines.
        if task.get("collinear"):
            z[:, 1] = z[:, 0] + 1e-6 * z[:, 1]
        kind = task["function"]
        if kind == "linear":
            y = 1.2 + 2.4 * z[:, 0] - 0.7 * z[:, 1] + 0.3 * z[:, 2]
        elif kind == "product":
            y = z[:, 0] * z[:, 1] + 0.6 * z[:, 2]
        elif kind == "sine":
            y = np.sin(2.3 * z[:, 0] + 0.4) + 0.2 * z[:, 1]
        elif kind == "decay":
            y = np.exp(-1.7 * z[:, 0]) + 0.3 * z[:, 1]
        elif kind == "logarithmic":
            y = np.log(2.2 + z[:, 0]) + z[:, 1] * z[:, 2]
        elif kind == "rational":
            y = (z[:, 0] + 0.4) / (1.5 + z[:, 1] ** 2)
        elif kind == "composed":
            y = np.sin(1.7 * z[:, 0] * z[:, 1] + 0.2) + np.tanh(z[:, 2] - z[:, 3])
        elif kind == "friedman1":
            u = (z + 1) / 2
            y = (
                10 * np.sin(np.pi * u[:, 0] * u[:, 1])
                + 20 * (u[:, 2] - 0.5) ** 2
                + 10 * u[:, 3]
                + 5 * u[:, 4]
            )
        else:
            raise ValueError(f"Unknown quality task function: {kind}")
        X = z * task.get("input_scale", 1) + task.get("input_offset", 0)
        return X.astype(np.float32).astype(float), y * task.get("target_scale", 1)

    X, y = values(train)
    Xt, yt = values(test)
    # Noise magnitude depends on training data only, never test target statistics.
    sigma = task.get("noise", 0) * np.std(y)
    y = y + rng.normal(0, sigma, len(y))
    yt = yt + test_rng.normal(0, sigma, len(yt))
    return X, y, Xt, yt

"""Fixed-candidate diagnostic adapters; numerical implementation lives in the package."""

import ctypes
from pathlib import Path

import numpy as np


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p) if a is not None else None


class Reference:
    def __init__(self):
        self.lib = ctypes.CDLL(
            str(Path(__file__).resolve().parents[2] / "build/scorer/reference.so")
        )
        self.lib.scorer_reference.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int] * 4
            + [ctypes.c_double]
            + [ctypes.c_void_p] * 3
        )
        self.lib.scorer_solve.argtypes = (
            [ctypes.c_void_p] * 2
            + [ctypes.c_int] * 4
            + [ctypes.c_double] * 2
            + [ctypes.c_void_p]
        )

    def fit(self, X, y, validation, descriptors, ridge, features=False):
        b, t, _ = descriptors.shape
        coef = np.empty((b, t + 1))
        losses = np.empty((b, 2))
        phi = np.empty((b, len(X), t)) if features else None
        rc = self.lib.scorer_reference(
            _ptr(X),
            _ptr(y),
            _ptr(validation),
            _ptr(descriptors),
            len(X),
            X.shape[1],
            b,
            t,
            ridge,
            _ptr(coef),
            _ptr(losses),
            _ptr(phi),
        )
        if rc:
            raise RuntimeError(f"C reference failed: {rc}")
        return coef, losses, phi

    def solve(self, factors, stats, terms, ntrain, ym, ridge):
        factors = np.ascontiguousarray(factors, dtype=np.float64)
        stats = np.ascontiguousarray(stats, dtype=np.float64)
        b, k, _ = factors.shape
        coef = np.empty((b, terms + 1))
        rc = self.lib.scorer_solve(
            _ptr(factors), _ptr(stats), b, terms, k, ntrain, ym, ridge, _ptr(coef)
        )
        if rc:
            raise RuntimeError(f"C reduced solve failed: {rc}")
        return coef


from wlearn_sym._family_scorer import FamilyScorer


class ResidentScorer(FamilyScorer):
    """Fixed-candidate diagnostic using the production public-frontend scorer."""

    def __init__(self, X, y, validation, descriptors, *, ridge=1e-8, **options):
        self.ref = Reference()
        terms = descriptors.shape[1]
        ntrain = int(np.count_nonzero(validation == 0))
        ym = float(y.astype(np.float32).astype(float)[validation == 0].mean())
        super().__init__(
            X,
            y,
            validation,
            descriptors,
            solve=lambda R, stats: self.ref.solve(R, stats, terms, ntrain, ym, ridge),
            target_mean=ym,
            ridge=ridge,
            **options,
        )

"""Numerical/lifecycle tests of the package scorer against the C reference."""

import numpy as np
import pytest

from resident import Reference, ResidentScorer, _ptr
from run import data
from search import Search


@pytest.mark.parametrize(
    "case", ["ordinary", "offset", "collinear", "constant", "protected"]
)
def test_reduced_c_solve_against_streaming_reference(case):
    X, y, mask, descriptors = data(96, 8, 3, case, 123)
    ref = Reference()
    coef, _, phi = ref.fit(X, y, mask, descriptors, 1e-8, features=True)
    anchor = phi[:, np.flatnonzero(mask == 0)[0] :][:, :1]
    delta = phi - anchor
    mean = delta[:, mask == 0].mean(axis=1, keepdims=True)
    centered = delta - mean
    scale = np.sqrt(np.mean(centered[:, mask == 0] ** 2, axis=1, keepdims=True))
    scale[scale == 0] = 1
    ym = y[mask == 0].mean()
    target = np.broadcast_to((y - ym)[None, :, None], (8, 96, 1))
    design = np.concatenate([centered / scale, target], axis=2)
    design[:, mask != 0] = 0
    factors = np.linalg.qr(design, mode="r")
    stats = np.concatenate([anchor, mean, scale], axis=2).reshape(8, 9)
    got = ref.solve(factors, stats, 3, np.count_nonzero(mask == 0), ym, 1e-8)
    expected = np.einsum("bnt,bt->bn", phi, coef[:, :3]) + coef[:, 3:]
    predicted = np.einsum("bnt,bt->bn", phi, got[:, :3]) + got[:, 3:]
    np.testing.assert_allclose(predicted, expected, atol=1e-5, rtol=1e-5)


def test_c_search_bridge_atomic_accept_and_polish():
    X, y, _, _ = data(128, 64, 3, "ordinary", 555)
    search = Search(X, y, 32, 3, 3, 44, 1)
    try:
        baseline = search.lib.scorer_fit(*search.args)
        expected = search.predict_and_free(baseline, X)
        assert not search.lib.sym_family_search_finish(search.handle)
        checked_rejection = False
        while count := search.next():
            coef, losses, _ = search.ref.fit(
                X,
                y,
                search.mask,
                np.ascontiguousarray(search.descriptors[:count]),
                1e-8,
            )
            if not checked_rejection:
                bad = losses.copy()
                bad[-1, -1] = np.nan
                assert (
                    search.lib.scorer_search_accept(
                        search.handle, search.id, _ptr(coef), _ptr(bad)
                    )
                    == -2
                )
            search.accept(coef, losses)
            if not checked_rejection:
                assert (
                    search.lib.scorer_search_accept(
                        search.handle, search.id, _ptr(coef), _ptr(losses)
                    )
                    == -1
                )
                checked_rejection = True
        model = search.lib.sym_family_search_finish(search.handle)
        actual = search.predict_and_free(model, X)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    finally:
        search.dispose()


@pytest.mark.parametrize("mode", ["custom", "tensor"])
def test_resident_precision_replay_and_lifetime(mode):
    X, y, mask, descriptors = data(64, 8, 4, "ordinary", 99)
    X[:, 0] += 1e6
    X[:, 1] = 1e-8
    X = X.astype(np.float32).astype(float)
    descriptors[:, :, 2] = np.arange(32).reshape(8, 4) % 10
    scorer = ResidentScorer(
        X,
        y,
        mask,
        descriptors,
        mode=mode,
        dtype="float64",
    )
    try:
        memories = []
        predictions = []
        for iteration in range(36):
            descriptors[:, :, 3] = (descriptors[:, :, 3] + 0.001).astype(np.float32)
            result = scorer.score(descriptors, inspect=True)
            coef, loss, phi = scorer.ref.fit(
                X, y, mask, descriptors, 1e-8, features=True
            )
            actual = (
                np.einsum("bnt,bt->bn", phi, result["coefficients"][:, :4])
                + result["coefficients"][:, 4:]
            )
            expected = np.einsum("bnt,bt->bn", phi, coef[:, :4]) + coef[:, 4:]
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(result["losses"], loss, atol=1e-5, rtol=1e-5)
            predictions.append(actual)
            if iteration:
                assert result["traffic"]["runtime_cache_misses"] == 0
                memories.append(scorer.rt.stats()["buffer_owned_current_bytes"])
            assert result["traffic"]["buffer_read_bytes"] == 8 * (3 * 4 + 8 * 8 + 2) * 8
        assert len(set(memories)) == 1
        assert not np.array_equal(predictions[0], predictions[-1])
        # Validation labels affect only reported validation loss, never readout fit.
        last = scorer.score(descriptors)
        changed = y.copy()
        changed[mask != 0] += 100
        scorer.y.copy_from(changed)
        changed_result = scorer.score(descriptors)
        np.testing.assert_array_equal(
            changed_result["coefficients"], last["coefficients"]
        )
        np.testing.assert_array_equal(
            changed_result["losses"][:, 0], last["losses"][:, 0]
        )
        assert np.all(changed_result["losses"][:, 1] > last["losses"][:, 1])
    finally:
        scorer.dispose()
    with pytest.raises((RuntimeError, ValueError)):
        scorer.score(descriptors)


def test_large_cosine_phase_is_rejected():
    X, y, mask, descriptors = data(64, 8, 1, "ordinary", 23)
    X[:] = np.linspace(1e15, 2e15, 64)[:, None]
    descriptors[:, :, 0] = 0
    descriptors[:, :, 2] = 5
    descriptors[:, :, 3] = 1
    descriptors[:, :, 4] = 0
    with pytest.raises(FloatingPointError, match="cosine phase"):
        ResidentScorer(
            X,
            y,
            mask,
            descriptors,
            dtype="float64",
        )

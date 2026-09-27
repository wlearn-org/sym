import ctypes as ct

import numpy as np
import pytest

from wlearn_sym._ffi import get_lib


def ptr(a):
    return a.ctypes.data_as(ct.POINTER(ct.c_double))


@pytest.mark.parametrize("classes", [0, 2, 3])
@pytest.mark.parametrize("polish", [0, 1])
def test_external_feature_tiles(classes, polish):
    lib = get_lib()
    r = np.arange(48)
    X = np.ascontiguousarray(np.column_stack([(r - 24) / 8, (r % 7 - 3) / 4]))
    y = np.asarray(r % classes if classes else r / 8 + (r % 7) / 16, dtype=float)
    config = np.array(
        [16, 3, 3, 4, 1, 4, 15, 42, polish, 0, 0.2, 1e-5, 0, 0.35, 0.55, 1e-8, 1e-12]
    )
    args = (ptr(X), 48, 2, ptr(y), int(classes > 0), classes, ptr(config), 17)
    direct = lib.wl_sym_family_fit(*args)
    state = lib.wl_sym_family_search_new(*args, 7, 17)
    stepped = None
    assert direct and state

    def raw(handle):
        out, length = ct.c_void_p(), ct.c_int()
        assert lib.sym_family_save(handle, ct.byref(out), ct.byref(length)) == 0
        try:
            return ct.string_at(out, length.value)
        finally:
            lib.wl_sym_free_buffer(out)

    try:
        while (status := lib.wl_sym_family_search_propose(state)) > 0:
            identity = lib.wl_sym_family_batch_id(state)
            shape = (ct.c_int * 8)()
            assert lib.wl_sym_family_batch_shape(state, shape, 8) == 0
            _, count, rows, cols, terms, start, _, _ = shape
            desc = np.empty((count, terms, 5))
            assert lib.wl_sym_family_batch_descriptors(state, ptr(desc), desc.size) == 0
            values = np.empty((count, rows, terms))
            for c in range(count):
                for t in range(terms):
                    a, b, op, p0, p1 = desc[c, t]
                    assert p0 == 1 and p1 == 0
                    x, yrow = (
                        X[start : start + rows, int(a)],
                        X[start : start + rows, int(b)],
                    )
                    divisor = np.where(
                        abs(yrow) < 1e-6, np.where(yrow >= 0, 1e-6, -1e-6), yrow
                    )
                    values[c, :, t] = [x + yrow, x - yrow, x * yrow, x / divisor][
                        int(op)
                    ]
            reference = np.empty_like(values)
            assert (
                lib.sym_family_search_score(
                    state, identity, ptr(reference), values.size
                )
                == 0
            )
            np.testing.assert_array_equal(values, reference)
            assert (
                lib.sym_family_search_accept(
                    state, identity + 1, ptr(values), values.size
                )
                == -1
            )
            assert (
                lib.sym_family_search_accept(state, identity, ptr(values), values.size)
                == 0
            )
        assert status == 0
        stepped = lib.sym_family_search_finish(state)
        assert stepped
        assert raw(stepped) == raw(direct)
    finally:
        lib.sym_family_search_free(state)
        lib.sym_family_free(direct)
        if stepped:
            lib.sym_family_free(stepped)

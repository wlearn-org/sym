"""Exercise the flattened frontend step ABI and its owned input buffers."""
import ctypes
import json
from pathlib import Path

import numpy as np

from wlearn_sym._ffi import get_lib


def test_step_abi_matches_fit_and_owns_training_buffers():
    lib = get_lib()
    fixture = Path(__file__).resolve().parents[2] / 'test' / 'search-abi.json'
    values = json.loads(fixture.read_text())['values']
    rows = np.arange(48, dtype=np.float64)
    X = np.ascontiguousarray(np.column_stack([np.sin(rows * .37), rows / 19]))
    y = np.ascontiguousarray(X[:, 0] * X[:, 1] + .2)
    ptr = ctypes.POINTER(ctypes.c_double)
    args = [X.ctypes.data_as(ptr), 48, 2, y.ctypes.data_as(ptr), *values]
    direct = lib.wl_sym_fit(*args)
    state = lib.wl_sym_search_new(*args)
    stepped = None
    assert direct and state
    try:
        X.fill(999)
        y.fill(999)
        scores = np.zeros(64 * 3, dtype=np.float64)
        batches = 0
        while True:
            count = lib.wl_sym_search_propose(state)
            assert count >= 0
            if count == 0:
                break
            batch_id = lib.wl_sym_search_batch_id(state)
            assert lib.wl_sym_search_score(state, batch_id, scores.ctypes.data_as(ptr), count) == 0
            assert lib.wl_sym_search_accept(state, batch_id, scores.ctypes.data_as(ptr), count) == 0
            batches += 1
        assert batches > 1
        stepped = lib.wl_sym_search_finish(state)
        assert stepped

        def raw(model):
            data = ctypes.c_void_p()
            size = ctypes.c_int()
            assert lib.wl_sym_save(model, ctypes.byref(data), ctypes.byref(size)) == 0
            try:
                return ctypes.string_at(data, size.value)
            finally:
                lib.wl_sym_free_buffer(data)

        assert raw(direct) == raw(stepped)
    finally:
        lib.wl_sym_search_free(state)
        lib.wl_sym_free(direct)
        if stepped:
            lib.wl_sym_free(stepped)

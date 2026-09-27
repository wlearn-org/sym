import ctypes
import ctypes.util
import importlib.machinery
import os

_lib = None

_D = ctypes.c_double
_I = ctypes.c_int
_DP = ctypes.POINTER(ctypes.c_double)


def _find_lib():
    env_path = os.environ.get("SYM_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    this_dir = os.path.dirname(os.path.abspath(__file__))
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = os.path.join(this_dir, "_native" + suffix)
        if os.path.isfile(candidate):
            return candidate

    dev_build = os.path.normpath(
        os.path.join(this_dir, "..", "..", "build", "libsym.so")
    )
    if os.path.isfile(dev_build):
        return dev_build

    return ctypes.util.find_library("sym")


def _declare(lib):
    lib.wl_sym_family_search_new.argtypes = [_DP, _I, _I, _DP, _I, _I, _DP, _I, _I, _I]
    lib.wl_sym_family_search_new.restype = ctypes.c_void_p
    lib.wl_sym_family_search_propose.argtypes = [ctypes.c_void_p]
    lib.wl_sym_family_search_propose.restype = _I
    lib.wl_sym_family_batch_id.argtypes = [ctypes.c_void_p]
    lib.wl_sym_family_batch_id.restype = ctypes.c_uint32
    lib.wl_sym_family_batch_shape.argtypes = [ctypes.c_void_p, ctypes.POINTER(_I), _I]
    lib.wl_sym_family_batch_shape.restype = _I
    lib.wl_sym_family_batch_descriptors.argtypes = [ctypes.c_void_p, _DP, _I]
    lib.wl_sym_family_batch_descriptors.restype = _I
    for name in ("sym_family_search_score", "sym_family_search_accept"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, _DP, _I]
        fn.restype = _I
    lib.sym_family_search_finish.argtypes = [ctypes.c_void_p]
    lib.sym_family_search_finish.restype = ctypes.c_void_p
    lib.sym_family_search_free.argtypes = [ctypes.c_void_p]
    lib.sym_family_search_free.restype = None
    lib.sym_family_save.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_I),
    ]
    lib.sym_family_save.restype = _I
    lib.sym_family_load.argtypes = [ctypes.c_void_p, _I]
    lib.sym_family_load.restype = ctypes.c_void_p
    lib.sym_family_dimensions.argtypes = [ctypes.c_void_p, ctypes.POINTER(_I)]
    lib.sym_family_dimensions.restype = _I
    lib.wl_sym_family_fit.argtypes = [_DP, _I, _I, _DP, _I, _I, _DP, _I]
    lib.wl_sym_family_fit.restype = ctypes.c_void_p
    lib.sym_family_new.argtypes = [_I, _I, _I, _I]
    lib.sym_family_new.restype = ctypes.c_void_p
    for name in ("sym_family_import", "sym_family_export"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, _I, _I, _DP, _I]
        fn.restype = _I
    lib.sym_family_archive_size.argtypes = [ctypes.c_void_p, _I]
    lib.sym_family_archive_size.restype = _I
    lib.sym_family_predict.argtypes = [ctypes.c_void_p, _DP, _I, _I, _I, _DP]
    lib.sym_family_predict.restype = _I
    lib.sym_family_free.argtypes = [ctypes.c_void_p]
    lib.sym_family_free.restype = None

    lib.wl_sym_get_last_error.argtypes = []
    lib.wl_sym_get_last_error.restype = ctypes.c_char_p

    lib.wl_sym_fit.argtypes = [
        _DP,
        _I,
        _I,
        _DP,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _D,
        _D,
        _D,
        _D,
        _D,
        _D,
        _D,
        _D,
        _D,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _I,
        _D,
    ]
    lib.wl_sym_fit.restype = ctypes.c_void_p

    lib.wl_sym_search_new.argtypes = lib.wl_sym_fit.argtypes
    lib.wl_sym_search_new.restype = ctypes.c_void_p
    lib.wl_sym_search_propose.argtypes = [ctypes.c_void_p]
    lib.wl_sym_search_propose.restype = _I
    lib.wl_sym_search_batch_id.argtypes = [ctypes.c_void_p]
    lib.wl_sym_search_batch_id.restype = ctypes.c_uint32
    for name in ("wl_sym_search_score", "wl_sym_search_accept"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, _DP, _I]
        fn.restype = _I
    lib.wl_sym_search_finish.argtypes = [ctypes.c_void_p]
    lib.wl_sym_search_finish.restype = ctypes.c_void_p
    lib.wl_sym_search_free.argtypes = [ctypes.c_void_p]
    lib.wl_sym_search_free.restype = None
    for name, restype in (
        ("stage", _I),
        ("rows", _I),
        ("row_indices", ctypes.POINTER(ctypes.c_int32)),
        ("X", _DP),
        ("target", _DP),
        ("mask", ctypes.POINTER(ctypes.c_uint8)),
    ):
        fn = getattr(lib, "wl_sym_search_batch_" + name)
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = restype
    lib.wl_sym_search_batch_node_count.argtypes = [ctypes.c_void_p, _I]
    lib.wl_sym_search_batch_node_count.restype = _I
    lib.wl_sym_search_batch_nodes.argtypes = [
        ctypes.c_void_p,
        _I,
        ctypes.POINTER(ctypes.c_int32),
        _DP,
        _I,
    ]
    lib.wl_sym_search_batch_nodes.restype = _I

    lib.wl_sym_predict.argtypes = [ctypes.c_void_p, _DP, _I, _I, _DP]
    lib.wl_sym_predict.restype = _I
    lib.wl_sym_predict_raw.argtypes = [ctypes.c_void_p, _DP, _I, _I, _DP]
    lib.wl_sym_predict_raw.restype = _I
    lib.wl_sym_predict_proba.argtypes = [ctypes.c_void_p, _DP, _I, _I, _DP]
    lib.wl_sym_predict_proba.restype = _I
    lib.wl_sym_transform.argtypes = [ctypes.c_void_p, _DP, _I, _I, _DP]
    lib.wl_sym_transform.restype = _I
    lib.wl_sym_score.argtypes = [ctypes.c_void_p, _DP, _I, _I, _DP]
    lib.wl_sym_score.restype = _D

    lib.wl_sym_save.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_I),
    ]
    lib.wl_sym_save.restype = _I
    lib.wl_sym_load.argtypes = [ctypes.c_void_p, _I]
    lib.wl_sym_load.restype = ctypes.c_void_p
    lib.wl_sym_free.argtypes = [ctypes.c_void_p]
    lib.wl_sym_free.restype = None
    lib.wl_sym_free_buffer.argtypes = [ctypes.c_void_p]
    lib.wl_sym_free_buffer.restype = None

    lib.wl_sym_formula_text.argtypes = [
        ctypes.c_void_p,
        _I,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_I),
    ]
    lib.wl_sym_formula_text.restype = _I
    lib.wl_sym_formula_json.argtypes = [
        ctypes.c_void_p,
        _I,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_I),
    ]
    lib.wl_sym_formula_json.restype = _I

    lib.wl_sym_set_formula_constant.argtypes = [ctypes.c_void_p, _I, _I, _D]
    lib.wl_sym_set_formula_constant.restype = _I
    lib.wl_sym_set_formula_metrics.argtypes = [ctypes.c_void_p, _I, _D, _D, _D]
    lib.wl_sym_set_formula_metrics.restype = _I

    lib.wl_sym_get_task.argtypes = [ctypes.c_void_p]
    lib.wl_sym_get_task.restype = _I
    lib.wl_sym_get_n_features.argtypes = [ctypes.c_void_p]
    lib.wl_sym_get_n_features.restype = _I
    lib.wl_sym_get_n_classes.argtypes = [ctypes.c_void_p]
    lib.wl_sym_get_n_classes.restype = _I
    lib.wl_sym_get_n_outputs.argtypes = [ctypes.c_void_p]
    lib.wl_sym_get_n_outputs.restype = _I
    lib.wl_sym_get_n_formulas.argtypes = [ctypes.c_void_p]
    lib.wl_sym_get_n_formulas.restype = _I
    lib.wl_sym_get_frontier_size.argtypes = [ctypes.c_void_p]
    lib.wl_sym_get_frontier_size.restype = _I
    lib.wl_sym_get_formula_n_nodes.argtypes = [ctypes.c_void_p, _I]
    lib.wl_sym_get_formula_n_nodes.restype = _I
    lib.wl_sym_get_formula_train_loss.argtypes = [ctypes.c_void_p, _I]
    lib.wl_sym_get_formula_train_loss.restype = _D
    lib.wl_sym_get_formula_valid_loss.argtypes = [ctypes.c_void_p, _I]
    lib.wl_sym_get_formula_valid_loss.restype = _D
    lib.wl_sym_get_formula_objective.argtypes = [ctypes.c_void_p, _I]
    lib.wl_sym_get_formula_objective.restype = _D
    lib.wl_sym_get_formula_complexity.argtypes = [ctypes.c_void_p, _I]
    lib.wl_sym_get_formula_complexity.restype = _D


def get_lib():
    global _lib
    if _lib is None:
        path = _find_lib()
        if path is None:
            raise RuntimeError(
                "Cannot find libsym. Set SYM_LIB_PATH, install wlearn-sym, "
                "or build with: cd sym && cmake -S . -B build && cmake --build build"
            )
        _lib = ctypes.CDLL(path)
        _declare(_lib)
    return _lib


def last_error(lib=None):
    lib = lib or get_lib()
    raw = lib.wl_sym_get_last_error()
    return raw.decode("utf-8", "replace") if raw else "unknown error"

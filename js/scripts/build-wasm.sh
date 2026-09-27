#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first."
  exit 1
fi

if [ -d "${PROJECT_DIR}/../src" ]; then
  node "${SCRIPT_DIR}/sync-csrc.js"
fi
mkdir -p "$OUTPUT_DIR"

EXPORTED_FUNCTIONS='[
  "_wl_sym_get_last_error",
  "_wl_sym_fit",
  "_wl_sym_family_fit",
  "_wl_sym_family_search_new",
  "_wl_sym_family_search_propose",
  "_wl_sym_family_batch_id",
  "_wl_sym_family_batch_shape",
  "_wl_sym_family_batch_descriptors",
  "_sym_family_search_score",
  "_sym_family_search_accept",
  "_sym_family_search_finish",
  "_sym_family_search_free",
  "_sym_family_new",
  "_sym_family_import",
  "_sym_family_export",
  "_sym_family_archive_size",
  "_sym_family_predict",
  "_sym_family_free",
  "_sym_family_save",
  "_sym_family_load",
  "_sym_family_dimensions",
  "_wl_sym_search_new",
  "_wl_sym_search_propose",
  "_wl_sym_search_batch_id",
  "_wl_sym_search_score",
  "_wl_sym_search_accept",
  "_wl_sym_search_finish",
  "_wl_sym_search_free",
  "_wl_sym_search_batch_stage",
  "_wl_sym_search_batch_rows",
  "_wl_sym_search_batch_row_indices",
  "_wl_sym_search_batch_X",
  "_wl_sym_search_batch_target",
  "_wl_sym_search_batch_mask",
  "_wl_sym_search_batch_node_count",
  "_wl_sym_search_batch_nodes",
  "_wl_sym_predict",
  "_wl_sym_predict_raw",
  "_wl_sym_predict_proba",
  "_wl_sym_transform",
  "_wl_sym_score",
  "_wl_sym_save",
  "_wl_sym_load",
  "_wl_sym_free",
  "_wl_sym_free_buffer",
  "_wl_sym_formula_text",
  "_wl_sym_formula_json",
  "_wl_sym_set_formula_constant",
  "_wl_sym_set_formula_metrics",
  "_wl_sym_get_task",
  "_wl_sym_get_n_features",
  "_wl_sym_get_n_classes",
  "_wl_sym_get_n_outputs",
  "_wl_sym_get_n_formulas",
  "_wl_sym_get_frontier_size",
  "_wl_sym_get_formula_n_nodes",
  "_wl_sym_get_formula_train_loss",
  "_wl_sym_get_formula_valid_loss",
  "_wl_sym_get_formula_objective",
  "_wl_sym_get_formula_complexity",
  "_malloc",
  "_free"
]'

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","HEAP32"]'

emcc \
  "${PROJECT_DIR}/csrc/sym.c" \
  "${PROJECT_DIR}/csrc/sym_tree_refine.c" \
  "${PROJECT_DIR}/csrc/sym_tree_io.c" \
  "${PROJECT_DIR}/csrc/sym_family.c" \
  "${PROJECT_DIR}/csrc/sym_family_io.c" \
  "${PROJECT_DIR}/csrc/wl_api.c" \
  -I "${PROJECT_DIR}/csrc" \
  -o "${OUTPUT_DIR}/sym.js" \
  -std=c11 \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s SINGLE_FILE_BINARY_ENCODE=0 \
  -s EXPORT_NAME=createSym \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=33554432 \
  -s ENVIRONMENT='web,node' \
  -O2

bash "${SCRIPT_DIR}/verify-exports.sh"

cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: none (C11 from scratch)
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 -std=c11 SINGLE_FILE=1
wasm_embedded: true
EOF

ls -lh "${OUTPUT_DIR}/sym.js"

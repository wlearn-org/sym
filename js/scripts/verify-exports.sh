#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WASM_FILE="${PROJECT_DIR}/wasm/sym.js"

if [ ! -f "$WASM_FILE" ]; then
  echo "ERROR: ${WASM_FILE} not found. Run build-wasm.sh first."
  exit 1
fi

EXPECTED_EXPORTS=(
  wl_sym_family_fit
  wl_sym_family_search_new
  wl_sym_family_search_propose
  wl_sym_family_batch_id
  wl_sym_family_batch_shape
  wl_sym_family_batch_descriptors
  sym_family_search_score
  sym_family_search_accept
  sym_family_search_data
  sym_family_search_solve
  sym_family_search_accept_results
  sym_family_search_residual_count
  sym_family_refine_data
  sym_family_refine_accept
  sym_family_search_finish
  sym_family_search_free
  sym_family_save
  sym_family_load
  sym_family_dimensions
  sym_family_new
  sym_family_import
  sym_family_export
  sym_family_archive_size
  sym_family_predict
  sym_family_free
  wl_sym_get_last_error
  wl_sym_fit
  wl_sym_predict
  wl_sym_predict_raw
  wl_sym_predict_proba
  wl_sym_transform
  wl_sym_score
  wl_sym_save
  wl_sym_load
  wl_sym_free
  wl_sym_free_buffer
  wl_sym_formula_text
  wl_sym_formula_json
  wl_sym_set_formula_constant
  wl_sym_set_formula_metrics
  wl_sym_get_task
  wl_sym_get_n_features
  wl_sym_get_n_classes
  wl_sym_get_n_outputs
  wl_sym_get_n_formulas
  wl_sym_get_frontier_size
  wl_sym_get_formula_n_nodes
  wl_sym_get_formula_train_loss
  wl_sym_get_formula_valid_loss
  wl_sym_get_formula_objective
  wl_sym_get_formula_complexity
)

missing=0
for fn in "${EXPECTED_EXPORTS[@]}"; do
  if ! grep -q "_${fn}" "$WASM_FILE"; then
    echo "MISSING: _${fn}"
    missing=$((missing + 1))
  fi
done

if [ "$missing" -gt 0 ]; then
  echo "ERROR: ${missing} exports missing from ${WASM_FILE}"
  exit 1
fi

echo "All ${#EXPECTED_EXPORTS[@]} exports verified."

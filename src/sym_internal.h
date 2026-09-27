#ifndef WLEARN_SYM_INTERNAL_H
#define WLEARN_SYM_INTERNAL_H
#include "sym.h"

/* Private cross-file helpers must not become interposable public ABI calls. */
#if defined(__GNUC__)
#pragma GCC visibility push(hidden)
#endif

/* The package shares one error channel across the tree and family modules. */
void sym_set_error(const char *message);

#define SYM_MAGIC "SYM1"
#define SYM_VERSION 1
#define SYM_EPS 1e-12
#define SYM_MAX_EXPR_RECURSION 512
#define SYM_LINEAR_BASIS_MAX 32
#define SYM_CONST_OPT_MAX 64

typedef struct {
    uint32_t state;
} sym_rng_t;
double sym_tree_clamp_double(double x, double lo, double hi);
int sym_tree_is_finite_value(double x);
const char *sym_tree_op_name(int32_t op);
int sym_tree_formula_init(sym_formula_t *f, int32_t capacity);
void sym_tree_formula_free(sym_formula_t *f);
int sym_tree_formula_copy(sym_formula_t *dst, const sym_formula_t *src);
int sym_tree_formula_add_node(sym_formula_t *f, sym_node_t node);
int sym_tree_formula_validate(const sym_formula_t *f, int32_t n_features);
int sym_tree_add_const_node(sym_formula_t *f, double value);
int sym_tree_add_op_node(sym_formula_t *f, int32_t op, int32_t left, int32_t right);
int sym_tree_formula_seed(sym_formula_t *f, int32_t seed_index, int32_t n_features,
                          const sym_params_t *params, double constant, const double *slopes,
                          const double *intercepts, const double *centers);
double sym_tree_formula_eval_row(const sym_formula_t *f, const double *row, int32_t n_features);
int sym_tree_collect_constant_nodes(const sym_formula_t *f, int32_t *nodes, int32_t max_nodes);
double sym_tree_formula_eval_row_jac_constants(const sym_formula_t *f, const double *row,
                                               int32_t n_features, const int32_t *const_nodes,
                                               int32_t n_const, double *jac);
int sym_tree_affine_fit_for_formula(const sym_formula_t *f, const double *X, const double *target,
                                    const uint8_t *is_valid, int32_t nrow, int32_t ncol,
                                    double *out_scale, double *out_bias);
void sym_tree_eval_formula_loss(sym_formula_t *f, const double *X, const double *target,
                                const uint8_t *is_valid, int32_t nrow, int32_t ncol,
                                const sym_params_t *params);
int sym_tree_formulas_equal_nodes(const sym_formula_t *a, const sym_formula_t *b);
void sym_tree_improve_constants(sym_formula_t *f, const double *X, const double *target,
                                const uint8_t *mask, int32_t nrow, int32_t ncol,
                                const sym_params_t *params);
void sym_tree_try_affine_rescale(sym_formula_t *f, const double *X, const double *target,
                                 const uint8_t *mask, int32_t nrow, int32_t ncol,
                                 const sym_params_t *params);
int sym_tree_try_linear_basis_combo(sym_formula_t *frontier, int32_t *n_frontier,
                                    int32_t frontier_cap, const double *X, const double *target,
                                    const uint8_t *mask, int32_t nrow, int32_t ncol,
                                    const sym_params_t *params, const double *seed_slopes,
                                    const double *seed_intercepts, const double *seed_centers,
                                    double seed_constant);
int sym_tree_add_to_frontier(sym_formula_t *frontier, int32_t *n_frontier, int32_t cap,
                             const sym_formula_t *cand);

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
#endif

#include "sym.h"
#include "sym_internal.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int solve_linear_system(double *A, double *b, double *x, int32_t n);

static int improve_constants_lm_mse(
    sym_formula_t *f,
    const double *X,
    const double *target,
    const uint8_t *mask,
    int32_t nrow,
    int32_t ncol,
    const sym_params_t *params,
    const int32_t *const_nodes,
    int32_t n_const
) {
    if (params->loss != SYM_LOSS_MSE || n_const <= 0 || n_const > SYM_CONST_OPT_MAX) return 0;
    int32_t n_train = 0;
    for (int32_t i = 0; i < nrow; i++) {
        if (!(mask && mask[i])) n_train++;
    }
    if (n_train <= n_const) return 0;

    double *A = (double *)malloc((size_t)n_const * (size_t)n_const * sizeof(double));
    double *Acopy = (double *)malloc((size_t)n_const * (size_t)n_const * sizeof(double));
    double *b = (double *)malloc((size_t)n_const * sizeof(double));
    double *bcopy = (double *)malloc((size_t)n_const * sizeof(double));
    double *delta = (double *)malloc((size_t)n_const * sizeof(double));
    double *jac = (double *)malloc((size_t)n_const * sizeof(double));
    double *base_values = (double *)malloc((size_t)n_const * sizeof(double));
    double *best_values = (double *)malloc((size_t)n_const * sizeof(double));
    if (!A || !Acopy || !b || !bcopy || !delta || !jac || !base_values || !best_values) {
        free(A);
        free(Acopy);
        free(b);
        free(bcopy);
        free(delta);
        free(jac);
        free(base_values);
        free(best_values);
        return 0;
    }

    double bound = fmax(fmax(fabs(params->const_min), fabs(params->const_max)) * 16.0, 64.0);
    int improved = 0;
    sym_tree_eval_formula_loss(f, X, target, mask, nrow, ncol, params);

    for (int iter = 0; iter < 10; iter++) {
        memset(A, 0, (size_t)n_const * (size_t)n_const * sizeof(double));
        memset(b, 0, (size_t)n_const * sizeof(double));
        for (int32_t j = 0; j < n_const; j++) base_values[j] = f->nodes[const_nodes[j]].value;

        for (int32_t i = 0; i < nrow; i++) {
            if (mask && mask[i]) continue;
            double pred = sym_tree_formula_eval_row_jac_constants(
                f, X + (size_t)i * ncol, ncol, const_nodes, n_const, jac);
            double residual = target[i] - pred;
            if (!sym_tree_is_finite_value(residual)) continue;
            for (int32_t j = 0; j < n_const; j++) {
                double jj = jac[j];
                if (!sym_tree_is_finite_value(jj)) continue;
                b[j] += jj * residual;
                for (int32_t k = j; k < n_const; k++) {
                    double jk = jac[k];
                    if (sym_tree_is_finite_value(jk)) A[(size_t)j * n_const + k] += jj * jk;
                }
            }
        }
        for (int32_t j = 0; j < n_const; j++) {
            for (int32_t k = 0; k < j; k++) A[(size_t)j * n_const + k] = A[(size_t)k * n_const + j];
        }

        double diag_mean = 0.0;
        double rhs_norm = 0.0;
        for (int32_t j = 0; j < n_const; j++) {
            diag_mean += fabs(A[(size_t)j * n_const + j]);
            rhs_norm += b[j] * b[j];
        }
        diag_mean = diag_mean > 0.0 ? diag_mean / (double)n_const : 1.0;
        if (rhs_norm <= 1e-20) break;

        const double lambdas[] = {1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0};
        double iter_best = f->objective;
        int accepted = 0;
        memcpy(best_values, base_values, (size_t)n_const * sizeof(double));
        for (int li = 0; li < 6; li++) {
            memcpy(Acopy, A, (size_t)n_const * (size_t)n_const * sizeof(double));
            memcpy(bcopy, b, (size_t)n_const * sizeof(double));
            for (int32_t j = 0; j < n_const; j++) {
                double d = fabs(A[(size_t)j * n_const + j]);
                Acopy[(size_t)j * n_const + j] += lambdas[li] * (d + diag_mean * 1e-6 + 1e-12);
            }
            if (solve_linear_system(Acopy, bcopy, delta, n_const) != 0) continue;

            double delta_norm = 0.0;
            for (int32_t j = 0; j < n_const; j++) delta_norm += delta[j] * delta[j];
            delta_norm = sqrt(delta_norm);
            double max_delta = bound * 0.5;
            double scale = (delta_norm > max_delta && delta_norm > 0.0) ? max_delta / delta_norm : 1.0;
            for (int32_t j = 0; j < n_const; j++) {
                double v = base_values[j] + scale * delta[j];
                f->nodes[const_nodes[j]].value = sym_tree_clamp_double(v, -bound, bound);
            }
            sym_tree_eval_formula_loss(f, X, target, mask, nrow, ncol, params);
            if (f->objective + 1e-12 < iter_best) {
                iter_best = f->objective;
                for (int32_t j = 0; j < n_const; j++) best_values[j] = f->nodes[const_nodes[j]].value;
                accepted = 1;
            }
            for (int32_t j = 0; j < n_const; j++) f->nodes[const_nodes[j]].value = base_values[j];
        }

        if (!accepted) {
            sym_tree_eval_formula_loss(f, X, target, mask, nrow, ncol, params);
            break;
        }
        for (int32_t j = 0; j < n_const; j++) f->nodes[const_nodes[j]].value = best_values[j];
        sym_tree_eval_formula_loss(f, X, target, mask, nrow, ncol, params);
        improved = 1;
    }

    free(A);
    free(Acopy);
    free(b);
    free(bcopy);
    free(delta);
    free(jac);
    free(base_values);
    free(best_values);
    return improved;
}

void sym_tree_improve_constants(
    sym_formula_t *f,
    const double *X,
    const double *target,
    const uint8_t *mask,
    int32_t nrow,
    int32_t ncol,
    const sym_params_t *params
) {
    int32_t const_nodes[SYM_CONST_OPT_MAX];
    int32_t n_const = sym_tree_collect_constant_nodes(f, const_nodes, SYM_CONST_OPT_MAX);
    if (n_const <= 0) {
        sym_tree_eval_formula_loss(f, X, target, mask, nrow, ncol, params);
        return;
    }
    improve_constants_lm_mse(f, X, target, mask, nrow, ncol, params, const_nodes, n_const);

    double best = f->objective;
    double span = fabs(params->const_max - params->const_min);
    double step0 = span > 0.0 ? span * 0.12 : 0.5;
    for (int pass = 0; pass < 3; pass++) {
        double step = step0 / (double)(pass + 1);
        for (int32_t cidx = 0; cidx < n_const; cidx++) {
            int32_t i = const_nodes[cidx];
            double old = f->nodes[i].value;
            double best_val = old;
            double candidates[4] = {old - step, old + step, old - 0.25 * step, old + 0.25 * step};
            for (int c = 0; c < 4; c++) {
                f->nodes[i].value = candidates[c];
                sym_tree_eval_formula_loss(f, X, target, mask, nrow, ncol, params);
                if (f->objective + 1e-15 < best) {
                    best = f->objective;
                    best_val = candidates[c];
                }
            }
            f->nodes[i].value = best_val;
            sym_tree_eval_formula_loss(f, X, target, mask, nrow, ncol, params);
        }
    }
}

void sym_tree_try_affine_rescale(
    sym_formula_t *f,
    const double *X,
    const double *target,
    const uint8_t *mask,
    int32_t nrow,
    int32_t ncol,
    const sym_params_t *params
) {
    if (!f || !f->nodes || f->n_nodes < 1 || f->n_nodes + 4 > f->capacity) return;
    if (params->loss != SYM_LOSS_MSE) return;

    double scale = 1.0;
    double bias = 0.0;
    if (!sym_tree_affine_fit_for_formula(f, X, target, mask, nrow, ncol, &scale, &bias)) return;
    if (fabs(scale - 1.0) < 1e-10 && fabs(bias) < 1e-10) return;

    double old_objective = f->objective;
    sym_formula_t tmp;
    if (sym_tree_formula_init(&tmp, f->capacity) != 0) return;
    memcpy(tmp.nodes, f->nodes, (size_t)f->n_nodes * sizeof(sym_node_t));
    tmp.n_nodes = f->n_nodes;
    int32_t root = tmp.n_nodes - 1;
    int32_t c_scale = sym_tree_add_const_node(&tmp, scale);
    int32_t mul = sym_tree_add_op_node(&tmp, SYM_OP_MUL, root, c_scale);
    int32_t c_bias = sym_tree_add_const_node(&tmp, bias);
    int32_t add = sym_tree_add_op_node(&tmp, SYM_OP_ADD, mul, c_bias);
    if (c_scale < 0 || mul < 0 || c_bias < 0 || add < 0) {
        sym_tree_formula_free(&tmp);
        return;
    }
    sym_tree_eval_formula_loss(&tmp, X, target, mask, nrow, ncol, params);
    if (tmp.objective <= old_objective + 1e-10 || tmp.valid_loss + 1e-15 < f->valid_loss) {
        sym_tree_formula_free(f);
        *f = tmp;
    } else {
        sym_tree_formula_free(&tmp);
    }
}

int sym_tree_add_to_frontier(sym_formula_t *frontier, int32_t *n_frontier, int32_t cap, const sym_formula_t *cand);

typedef struct {
    sym_formula_t formula;
    double score;
} sym_basis_item_t;

static double formula_abs_corr(
    const sym_formula_t *f,
    const double *X,
    const double *target,
    const uint8_t *mask,
    int32_t nrow,
    int32_t ncol
) {
    double sum_p = 0.0;
    double sum_y = 0.0;
    int32_t n = 0;
    for (int32_t i = 0; i < nrow; i++) {
        if (mask && mask[i]) continue;
        double p = sym_tree_formula_eval_row(f, X + (size_t)i * ncol, ncol);
        sum_p += p;
        sum_y += target[i];
        n++;
    }
    if (n <= 1) return 0.0;
    double mean_p = sum_p / (double)n;
    double mean_y = sum_y / (double)n;
    double var_p = 0.0;
    double var_y = 0.0;
    double cov = 0.0;
    for (int32_t i = 0; i < nrow; i++) {
        if (mask && mask[i]) continue;
        double p = sym_tree_formula_eval_row(f, X + (size_t)i * ncol, ncol);
        double dp = p - mean_p;
        double dy = target[i] - mean_y;
        var_p += dp * dp;
        var_y += dy * dy;
        cov += dp * dy;
    }
    if (var_p <= SYM_EPS || var_y <= SYM_EPS) return 0.0;
    double corr = fabs(cov / sqrt(var_p * var_y));
    return sym_tree_is_finite_value(corr) ? corr : 0.0;
}

static int basis_pool_add(
    sym_basis_item_t *basis,
    int32_t *n_basis,
    int32_t cap,
    const sym_formula_t *cand,
    double score
) {
    if (!cand || !cand->nodes || cand->n_nodes <= 0 || score <= 1e-12) return 0;
    for (int32_t i = 0; i < *n_basis; i++) {
        if (sym_tree_formulas_equal_nodes(&basis[i].formula, cand)) {
            if (score > basis[i].score) basis[i].score = score;
            return 0;
        }
    }

    int32_t slot = -1;
    if (*n_basis < cap) {
        slot = *n_basis;
        (*n_basis)++;
    } else {
        double worst = DBL_MAX;
        int32_t worst_i = -1;
        for (int32_t i = 0; i < *n_basis; i++) {
            if (basis[i].score < worst) {
                worst = basis[i].score;
                worst_i = i;
            }
        }
        if (score <= worst) return 0;
        slot = worst_i;
    }

    sym_tree_formula_free(&basis[slot].formula);
    if (sym_tree_formula_copy(&basis[slot].formula, cand) != 0) return -1;
    basis[slot].score = score;
    return 0;
}

static int append_formula_nodes(sym_formula_t *dst, const sym_formula_t *src, int32_t *out_root) {
    if (!dst || !src || !src->nodes || src->n_nodes <= 0 || !out_root) return -1;
    if (dst->n_nodes + src->n_nodes > dst->capacity) return -1;
    int32_t offset = dst->n_nodes;
    for (int32_t i = 0; i < src->n_nodes; i++) {
        sym_node_t n = src->nodes[i];
        if (n.left >= 0) n.left += offset;
        if (n.right >= 0) n.right += offset;
        if (sym_tree_formula_add_node(dst, n) < 0) return -1;
    }
    *out_root = offset + src->n_nodes - 1;
    return 0;
}

static int solve_linear_system(double *A, double *b, double *x, int32_t n) {
    for (int32_t i = 0; i < n; i++) x[i] = 0.0;
    for (int32_t col = 0; col < n; col++) {
        int32_t pivot = col;
        double best = fabs(A[(size_t)col * n + col]);
        for (int32_t r = col + 1; r < n; r++) {
            double v = fabs(A[(size_t)r * n + col]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (best <= 1e-14) return -1;
        if (pivot != col) {
            for (int32_t c = col; c < n; c++) {
                double t = A[(size_t)col * n + c];
                A[(size_t)col * n + c] = A[(size_t)pivot * n + c];
                A[(size_t)pivot * n + c] = t;
            }
            double tb = b[col];
            b[col] = b[pivot];
            b[pivot] = tb;
        }
        double diag = A[(size_t)col * n + col];
        for (int32_t c = col; c < n; c++) A[(size_t)col * n + c] /= diag;
        b[col] /= diag;
        for (int32_t r = 0; r < n; r++) {
            if (r == col) continue;
            double factor = A[(size_t)r * n + col];
            if (fabs(factor) <= 1e-18) continue;
            for (int32_t c = col; c < n; c++) {
                A[(size_t)r * n + c] -= factor * A[(size_t)col * n + c];
            }
            b[r] -= factor * b[col];
        }
    }
    for (int32_t i = 0; i < n; i++) x[i] = b[i];
    return 0;
}

static int fit_linear_basis(
    const double *basis_pred,
    int32_t n_basis,
    const double *target,
    const uint8_t *mask,
    int32_t nrow,
    double ridge,
    double *coef
) {
    int32_t d = n_basis + 1;
    double A[(SYM_LINEAR_BASIS_MAX + 1) * (SYM_LINEAR_BASIS_MAX + 1)];
    double b[SYM_LINEAR_BASIS_MAX + 1];
    memset(A, 0, sizeof(A));
    memset(b, 0, sizeof(b));
    int32_t n_train = 0;
    for (int32_t i = 0; i < nrow; i++) {
        if (mask && mask[i]) continue;
        n_train++;
        double y = target[i];
        b[0] += y;
        A[0] += 1.0;
        for (int32_t j = 0; j < n_basis; j++) {
            double zj = basis_pred[(size_t)j * nrow + i];
            b[j + 1] += zj * y;
            A[j + 1] += zj;
            A[(size_t)(j + 1) * d] += zj;
            for (int32_t k = j; k < n_basis; k++) {
                double zk = basis_pred[(size_t)k * nrow + i];
                A[(size_t)(j + 1) * d + (k + 1)] += zj * zk;
            }
        }
    }
    if (n_train <= d) return -1;
    for (int32_t j = 1; j <= n_basis; j++) {
        for (int32_t k = 1; k < j; k++) {
            A[(size_t)j * d + k] = A[(size_t)k * d + j];
        }
        A[(size_t)j * d + j] += ridge * (double)n_train;
    }
    return solve_linear_system(A, b, coef, d);
}

typedef struct {
    int32_t idx;
    double contribution;
} sym_term_rank_t;

static void sort_term_ranks(sym_term_rank_t *items, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = i + 1; j < n; j++) {
            if (items[j].contribution > items[i].contribution) {
                sym_term_rank_t t = items[i];
                items[i] = items[j];
                items[j] = t;
            }
        }
    }
}

static double prediction_abs_corr(
    const double *a,
    const double *b,
    const uint8_t *mask,
    int32_t nrow
) {
    double sum_a = 0.0;
    double sum_b = 0.0;
    int32_t n = 0;
    for (int32_t i = 0; i < nrow; i++) {
        if (mask && mask[i]) continue;
        sum_a += a[i];
        sum_b += b[i];
        n++;
    }
    if (n <= 1) return 0.0;
    double mean_a = sum_a / (double)n;
    double mean_b = sum_b / (double)n;
    double va = 0.0;
    double vb = 0.0;
    double cov = 0.0;
    for (int32_t i = 0; i < nrow; i++) {
        if (mask && mask[i]) continue;
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        va += da * da;
        vb += db * db;
        cov += da * db;
    }
    if (va <= SYM_EPS || vb <= SYM_EPS) return 0.0;
    double corr = fabs(cov / sqrt(va * vb));
    return sym_tree_is_finite_value(corr) ? corr : 0.0;
}

int sym_tree_try_linear_basis_combo(
    sym_formula_t *frontier,
    int32_t *n_frontier,
    int32_t frontier_cap,
    const double *X,
    const double *target,
    const uint8_t *mask,
    int32_t nrow,
    int32_t ncol,
    const sym_params_t *params,
    const double *seed_slopes,
    const double *seed_intercepts,
    const double *seed_centers,
    double seed_constant
) {
    if (!frontier || !n_frontier || *n_frontier <= 0 || params->loss != SYM_LOSS_MSE) return 0;
    if (params->max_nodes < 5) return 0;

    sym_basis_item_t basis[SYM_LINEAR_BASIS_MAX];
    memset(basis, 0, sizeof(basis));
    int32_t n_basis = 0;
    double *basis_pred = NULL;

    for (int32_t i = 0; i < *n_frontier && i < params->frontier_size; i++) {
        double score = formula_abs_corr(&frontier[i], X, target, mask, nrow, ncol);
        if (basis_pool_add(basis, &n_basis, SYM_LINEAR_BASIS_MAX, &frontier[i], score) != 0) goto fail;
    }

    int64_t pair_count64 = (int64_t)ncol * (int64_t)ncol;
    int64_t seed_scan = 3 + (int64_t)ncol * 3 + 1;
    if (pair_count64 > 0 && pair_count64 < INT32_MAX / 8) {
        if (params->operator_set >= SYM_OPSET_SMOOTH) seed_scan += pair_count64;
        seed_scan += pair_count64 * 4;
    }
    int64_t max_scan = (int64_t)params->population * 4;
    if (max_scan < 128) max_scan = 128;
    if (max_scan > 1024) max_scan = 1024;
    if (seed_scan > max_scan) seed_scan = max_scan;

    for (int64_t si = 0; si < seed_scan; si++) {
        sym_formula_t seed_formula;
        memset(&seed_formula, 0, sizeof(seed_formula));
        int seeded = sym_tree_formula_seed(&seed_formula, (int32_t)si, ncol, params, seed_constant,
                                  seed_slopes, seed_intercepts, seed_centers);
        if (seeded < 0) goto fail;
        if (!seeded) continue;
        double score = formula_abs_corr(&seed_formula, X, target, mask, nrow, ncol);
        int rc = basis_pool_add(basis, &n_basis, SYM_LINEAR_BASIS_MAX, &seed_formula, score);
        sym_tree_formula_free(&seed_formula);
        if (rc != 0) goto fail;
    }

    if (n_basis <= 0) goto done;
    basis_pred = (double *)malloc((size_t)n_basis * (size_t)nrow * sizeof(double));
    if (!basis_pred) goto fail;
    for (int32_t j = 0; j < n_basis; j++) {
        for (int32_t i = 0; i < nrow; i++) {
            basis_pred[(size_t)j * nrow + i] = sym_tree_formula_eval_row(
                &basis[j].formula, X + (size_t)i * ncol, ncol);
        }
    }

    double coef[SYM_LINEAR_BASIS_MAX + 1];
    if (fit_linear_basis(basis_pred, n_basis, target, mask, nrow, 1e-8, coef) != 0) {
        free(basis_pred);
        basis_pred = NULL;
        goto done;
    }

    sym_term_rank_t ranks[SYM_LINEAR_BASIS_MAX];
    for (int32_t j = 0; j < n_basis; j++) {
        double mean = 0.0;
        double sq = 0.0;
        int32_t n = 0;
        for (int32_t i = 0; i < nrow; i++) {
            if (mask && mask[i]) continue;
            double v = coef[j + 1] * basis_pred[(size_t)j * nrow + i];
            mean += v;
            sq += v * v;
            n++;
        }
        double contrib = fabs(coef[j + 1]);
        if (n > 1) {
            mean /= (double)n;
            double var = sq / (double)n - mean * mean;
            if (var > 0.0 && sym_tree_is_finite_value(var)) contrib = sqrt(var);
        }
        ranks[j].idx = j;
        ranks[j].contribution = sym_tree_is_finite_value(contrib) ? contrib : 0.0;
    }
    sort_term_ranks(ranks, n_basis);

    sym_formula_t combo;
    if (sym_tree_formula_init(&combo, params->max_nodes) != 0) goto fail;
    int32_t root = sym_tree_add_const_node(&combo, coef[0]);
    if (root < 0) {
        sym_tree_formula_free(&combo);
        goto fail;
    }
    int32_t selected = 0;
    int32_t selected_basis[SYM_LINEAR_BASIS_MAX];
    int32_t max_terms = params->top_k;
    if (max_terms > 12) max_terms = 12;
    for (int32_t r = 0; r < n_basis && selected < max_terms; r++) {
        int32_t j = ranks[r].idx;
        double w = coef[j + 1];
        if (fabs(w) < 1e-10 || ranks[r].contribution <= 1e-12) continue;
        int too_correlated = 0;
        for (int32_t s = 0; s < selected; s++) {
            int32_t prev = selected_basis[s];
            double corr = prediction_abs_corr(
                basis_pred + (size_t)j * nrow,
                basis_pred + (size_t)prev * nrow,
                mask,
                nrow
            );
            if (corr > 0.995) {
                too_correlated = 1;
                break;
            }
        }
        if (too_correlated) continue;
        if (combo.n_nodes + basis[j].formula.n_nodes + 3 > combo.capacity) continue;
        int32_t subroot = -1;
        if (append_formula_nodes(&combo, &basis[j].formula, &subroot) != 0) continue;
        int32_t cw = sym_tree_add_const_node(&combo, w);
        int32_t mul = sym_tree_add_op_node(&combo, SYM_OP_MUL, subroot, cw);
        int32_t add = sym_tree_add_op_node(&combo, SYM_OP_ADD, root, mul);
        if (cw < 0 || mul < 0 || add < 0) break;
        root = add;
        selected_basis[selected] = j;
        selected++;
    }
    if (selected > 0 && sym_tree_formula_validate(&combo, ncol) == 0) {
        sym_tree_eval_formula_loss(&combo, X, target, mask, nrow, ncol, params);
        sym_tree_improve_constants(&combo, X, target, mask, nrow, ncol, params);
        sym_tree_add_to_frontier(frontier, n_frontier, frontier_cap, &combo);
    }
    sym_tree_formula_free(&combo);

    int32_t omp_max_terms = params->top_k;
    if (omp_max_terms > 12) omp_max_terms = 12;
    if (omp_max_terms > n_basis) omp_max_terms = n_basis;
    double *residual = NULL;
    double *selected_pred = NULL;
    if (omp_max_terms > 0) {
        residual = (double *)malloc((size_t)nrow * sizeof(double));
        selected_pred = (double *)malloc((size_t)omp_max_terms * (size_t)nrow * sizeof(double));
    }
    if (residual && selected_pred) {
        double mean_y = 0.0;
        int32_t n_train = 0;
        for (int32_t i = 0; i < nrow; i++) {
            if (mask && mask[i]) continue;
            mean_y += target[i];
            n_train++;
        }
        mean_y = n_train > 0 ? mean_y / (double)n_train : 0.0;
        for (int32_t i = 0; i < nrow; i++) residual[i] = target[i] - mean_y;

        int32_t omp_selected[SYM_LINEAR_BASIS_MAX];
        int32_t omp_n = 0;
        int32_t used_nodes = 1;
        double omp_coef[SYM_LINEAR_BASIS_MAX + 1];
        memset(omp_coef, 0, sizeof(omp_coef));
        omp_coef[0] = mean_y;

        for (int32_t step = 0; step < omp_max_terms; step++) {
            int32_t best_j = -1;
            double best_score = 1e-10;
            for (int32_t j = 0; j < n_basis; j++) {
                int already = 0;
                for (int32_t s = 0; s < omp_n; s++) {
                    if (omp_selected[s] == j) {
                        already = 1;
                        break;
                    }
                }
                if (already) continue;
                if (used_nodes + basis[j].formula.n_nodes + 3 > params->max_nodes) continue;

                int too_correlated = 0;
                for (int32_t s = 0; s < omp_n; s++) {
                    int32_t prev = omp_selected[s];
                    double corr = prediction_abs_corr(
                        basis_pred + (size_t)j * nrow,
                        basis_pred + (size_t)prev * nrow,
                        mask,
                        nrow
                    );
                    if (corr > 0.995) {
                        too_correlated = 1;
                        break;
                    }
                }
                if (too_correlated) continue;

                double score = prediction_abs_corr(
                    basis_pred + (size_t)j * nrow,
                    residual,
                    mask,
                    nrow
                );
                if (score > best_score) {
                    best_score = score;
                    best_j = j;
                }
            }
            if (best_j < 0) break;

            omp_selected[omp_n++] = best_j;
            used_nodes += basis[best_j].formula.n_nodes + 3;
            for (int32_t s = 0; s < omp_n; s++) {
                int32_t j = omp_selected[s];
                memcpy(selected_pred + (size_t)s * nrow,
                       basis_pred + (size_t)j * nrow,
                       (size_t)nrow * sizeof(double));
            }
            double coef_work[SYM_LINEAR_BASIS_MAX + 1];
            if (fit_linear_basis(selected_pred, omp_n, target, mask, nrow, 1e-8, coef_work) != 0) {
                omp_n--;
                used_nodes -= basis[best_j].formula.n_nodes + 3;
                break;
            }
            memcpy(omp_coef, coef_work, (size_t)(omp_n + 1) * sizeof(double));
            for (int32_t i = 0; i < nrow; i++) {
                double pred = omp_coef[0];
                for (int32_t s = 0; s < omp_n; s++) {
                    pred += omp_coef[s + 1] * selected_pred[(size_t)s * nrow + i];
                }
                residual[i] = target[i] - pred;
            }
        }

        if (omp_n > 0) {
            sym_formula_t omp_combo;
            if (sym_tree_formula_init(&omp_combo, params->max_nodes) == 0) {
                int32_t omp_root = sym_tree_add_const_node(&omp_combo, omp_coef[0]);
                int32_t emitted = 0;
                if (omp_root >= 0) {
                    for (int32_t s = 0; s < omp_n; s++) {
                        int32_t j = omp_selected[s];
                        double w = omp_coef[s + 1];
                        if (fabs(w) < 1e-10) continue;
                        if (omp_combo.n_nodes + basis[j].formula.n_nodes + 3 > omp_combo.capacity) continue;
                        int32_t subroot = -1;
                        if (append_formula_nodes(&omp_combo, &basis[j].formula, &subroot) != 0) continue;
                        int32_t cw = sym_tree_add_const_node(&omp_combo, w);
                        int32_t mul = sym_tree_add_op_node(&omp_combo, SYM_OP_MUL, subroot, cw);
                        int32_t add = sym_tree_add_op_node(&omp_combo, SYM_OP_ADD, omp_root, mul);
                        if (cw < 0 || mul < 0 || add < 0) break;
                        omp_root = add;
                        emitted++;
                    }
                }
                if (emitted > 0 && sym_tree_formula_validate(&omp_combo, ncol) == 0) {
                    sym_tree_eval_formula_loss(&omp_combo, X, target, mask, nrow, ncol, params);
                    sym_tree_improve_constants(&omp_combo, X, target, mask, nrow, ncol, params);
                    sym_tree_add_to_frontier(frontier, n_frontier, frontier_cap, &omp_combo);
                }
                sym_tree_formula_free(&omp_combo);
            }
        }
    }
    free(residual);
    free(selected_pred);
    free(basis_pred);
    basis_pred = NULL;

done:
    free(basis_pred);
    for (int32_t i = 0; i < n_basis; i++) sym_tree_formula_free(&basis[i].formula);
    return 0;

fail:
    free(basis_pred);
    for (int32_t i = 0; i < n_basis; i++) sym_tree_formula_free(&basis[i].formula);
    return -1;
}

#include "sym.h"
#include "sym_internal.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static char g_error[512] = "";

void sym_set_error(const char *msg) {
    if (!msg) msg = "unknown error";
    snprintf(g_error, sizeof(g_error), "%s", msg);
}


const char *sym_get_error(void) {
    return g_error;
}

void sym_free_buffer(void *ptr) {
    free(ptr);
}


static uint32_t rng_next(sym_rng_t *rng) {
    rng->state = (rng->state * 1664525u + 1013904223u) & 0x7FFFFFFFu;
    return rng->state;
}

static double rng_uniform(sym_rng_t *rng) {
    return (double)rng_next(rng) / (double)0x7FFFFFFFu;
}

static int32_t rng_int(sym_rng_t *rng, int32_t n) {
    if (n <= 1) return 0;
    int32_t v = (int32_t)(rng_uniform(rng) * (double)n);
    return v >= n ? n - 1 : v;
}

static double rng_normalish(sym_rng_t *rng) {
    double u = rng_uniform(rng) - 0.5;
    double v = rng_uniform(rng) - 0.5;
    double w = rng_uniform(rng) - 0.5;
    return (u + v + w) * 2.0;
}

double sym_tree_clamp_double(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

int sym_tree_is_finite_value(double x) {
    return isfinite(x) && !isnan(x);
}

const char *sym_tree_op_name(int32_t op) {
    switch (op) {
        case SYM_OP_CONST: return "const";
        case SYM_OP_VAR: return "var";
        case SYM_OP_ADD: return "add";
        case SYM_OP_SUB: return "sub";
        case SYM_OP_MUL: return "mul";
        case SYM_OP_DIV: return "div";
        case SYM_OP_NEG: return "neg";
        case SYM_OP_ABS: return "abs";
        case SYM_OP_SQRT: return "sqrt";
        case SYM_OP_LOG: return "log";
        case SYM_OP_EXP: return "exp";
        case SYM_OP_SIN: return "sin";
        case SYM_OP_COS: return "cos";
        case SYM_OP_TANH: return "tanh";
        case SYM_OP_MIN: return "min";
        case SYM_OP_MAX: return "max";
        default: return "unknown";
    }
}

static int32_t op_arity(int32_t op) {
    switch (op) {
        case SYM_OP_CONST:
        case SYM_OP_VAR:
            return 0;
        case SYM_OP_NEG:
        case SYM_OP_ABS:
        case SYM_OP_SQRT:
        case SYM_OP_LOG:
        case SYM_OP_EXP:
        case SYM_OP_SIN:
        case SYM_OP_COS:
        case SYM_OP_TANH:
            return 1;
        default:
            return 2;
    }
}

static double op_cost(int32_t op) {
    switch (op) {
        case SYM_OP_CONST:
        case SYM_OP_VAR:
            return 1.0;
        case SYM_OP_ADD:
        case SYM_OP_SUB:
            return 1.0;
        case SYM_OP_MUL:
        case SYM_OP_DIV:
        case SYM_OP_NEG:
        case SYM_OP_ABS:
        case SYM_OP_MIN:
        case SYM_OP_MAX:
            return 2.0;
        case SYM_OP_SQRT:
        case SYM_OP_LOG:
        case SYM_OP_EXP:
        case SYM_OP_SIN:
        case SYM_OP_COS:
        case SYM_OP_TANH:
            return 3.0;
        default:
            return 10.0;
    }
}

void sym_params_init(sym_params_t *params) {
    if (!params) return;
    memset(params, 0, sizeof(*params));
    params->task = SYM_TASK_REGRESSION;
    params->population = 256;
    params->generations = 120;
    params->max_nodes = 31;
    params->max_depth = 6;
    params->frontier_size = 16;
    params->tournament_size = 4;
    params->elite_count = 4;
    params->top_k = 8;
    params->n_classes = 0;
    params->loss = SYM_LOSS_MSE;
    params->operator_set = SYM_OPSET_FULL;
    params->early_stop_rounds = 30;
    params->seed = 42u;
    params->validation_fraction = 0.0;
    params->complexity_penalty = 0.001;
    params->mutation_rate = 0.35;
    params->crossover_rate = 0.55;
    params->constant_rate = 0.20;
    params->const_min = -4.0;
    params->const_max = 4.0;
    params->huber_delta = 1.0;
    params->tol = 1e-12;
    params->islands = 1;
    params->migration_interval = 0;
    params->migration_count = 0;
    params->warmup_generations = 0;
    params->warmup_min_nodes = 0;
    params->brood_size = 1;
    params->row_sample_size = 0;
    params->local_refine_interval = 0;
    params->local_refine_count = 0;
    params->complexity_hof_size = 0;
    params->final_selector = SYM_FINAL_OBJECTIVE;
    params->complexity_bucket_width = 2.0;
}

static void normalize_params(sym_params_t *p, int32_t nrow, int32_t ncol) {
    if (p->population < 8) p->population = 8;
    if (p->population > 20000) p->population = 20000;
    if (p->generations < 1) p->generations = 1;
    if (p->max_nodes < 1) p->max_nodes = 1;
    if (p->max_nodes > 255) p->max_nodes = 255;
    if ((p->max_nodes & 1) == 0) p->max_nodes -= 1;
    if (p->max_depth < 1) p->max_depth = 1;
    if (p->max_depth > p->max_nodes) p->max_depth = p->max_nodes;
    if (p->frontier_size < 1) p->frontier_size = 1;
    if (p->frontier_size > 128) p->frontier_size = 128;
    if (p->tournament_size < 2) p->tournament_size = 2;
    if (p->tournament_size > p->population) p->tournament_size = p->population;
    if (p->elite_count < 1) p->elite_count = 1;
    if (p->elite_count > p->population / 2) p->elite_count = p->population / 2;
    if (p->top_k < 1) p->top_k = 1;
    if (p->top_k > p->frontier_size) p->top_k = p->frontier_size;
    if (p->validation_fraction < 0.0) p->validation_fraction = 0.0;
    if (p->validation_fraction > 0.8) p->validation_fraction = 0.8;
    if (nrow < 5) p->validation_fraction = 0.0;
    if (p->complexity_penalty < 0.0) p->complexity_penalty = 0.0;
    p->mutation_rate = sym_tree_clamp_double(p->mutation_rate, 0.0, 1.0);
    p->crossover_rate = sym_tree_clamp_double(p->crossover_rate, 0.0, 1.0);
    p->constant_rate = sym_tree_clamp_double(p->constant_rate, 0.0, 1.0);
    if (p->const_min > p->const_max) {
        double t = p->const_min;
        p->const_min = p->const_max;
        p->const_max = t;
    }
    if (p->huber_delta <= 0.0) p->huber_delta = 1.0;
    if (p->tol <= 0.0) p->tol = 1e-12;
    if (p->operator_set < SYM_OPSET_BASIC || p->operator_set > SYM_OPSET_FULL) {
        p->operator_set = SYM_OPSET_FULL;
    }
    if (p->task == SYM_TASK_CLASSIFICATION && p->n_classes < 2) p->n_classes = 2;
    if (p->task != SYM_TASK_CLASSIFICATION) p->n_classes = 0;
    if (ncol < 1) p->max_nodes = 1;
    if (p->islands < 1) p->islands = 1;
    if (p->islands > p->population) p->islands = p->population;
    if (p->migration_interval < 0) p->migration_interval = 0;
    if (p->migration_count < 0) p->migration_count = 0;
    int32_t min_island = p->population / p->islands;
    if (min_island < 1) min_island = 1;
    if (p->migration_count > min_island / 2) p->migration_count = min_island / 2;
    if (p->warmup_generations < 0) p->warmup_generations = 0;
    if (p->warmup_generations > p->generations) p->warmup_generations = p->generations;
    if (p->warmup_min_nodes < 1) p->warmup_min_nodes = 1;
    if (p->warmup_min_nodes > p->max_nodes) p->warmup_min_nodes = p->max_nodes;
    if ((p->warmup_min_nodes & 1) == 0) p->warmup_min_nodes -= 1;
    if (p->warmup_min_nodes < 1) p->warmup_min_nodes = 1;
    if (p->brood_size < 1) p->brood_size = 1;
    if (p->brood_size > 16) p->brood_size = 16;
    if (p->row_sample_size < 0) p->row_sample_size = 0;
    if (p->row_sample_size >= nrow) p->row_sample_size = 0;
    if (p->row_sample_size > 0 && p->row_sample_size < 16 && nrow >= 16) p->row_sample_size = 16;
    if (p->local_refine_interval < 0) p->local_refine_interval = 0;
    if (p->local_refine_count < 0) p->local_refine_count = 0;
    if (p->local_refine_count > 32) p->local_refine_count = 32;
    if (p->local_refine_count > 0 && p->local_refine_interval <= 0) p->local_refine_interval = 10;
    if (p->complexity_hof_size < 0) p->complexity_hof_size = 0;
    if (p->complexity_hof_size > 512) p->complexity_hof_size = 512;
    if (p->final_selector < SYM_FINAL_OBJECTIVE || p->final_selector > SYM_FINAL_SCORE) {
        p->final_selector = SYM_FINAL_OBJECTIVE;
    }
    if (p->complexity_bucket_width <= 0.0 || !sym_tree_is_finite_value(p->complexity_bucket_width)) {
        p->complexity_bucket_width = 2.0;
    }
    if (p->complexity_bucket_width < 0.25) p->complexity_bucket_width = 0.25;
    if (p->complexity_bucket_width > 32.0) p->complexity_bucket_width = 32.0;
}

int sym_tree_formula_init(sym_formula_t *f, int32_t capacity) {
    if (!f || capacity <= 0) return -1;
    memset(f, 0, sizeof(*f));
    f->nodes = (sym_node_t *)calloc((size_t)capacity, sizeof(sym_node_t));
    if (!f->nodes) {
        sym_set_error("out of memory allocating formula nodes");
        return -1;
    }
    f->capacity = capacity;
    f->train_loss = DBL_MAX;
    f->valid_loss = DBL_MAX;
    f->objective = DBL_MAX;
    f->complexity = 0.0;
    return 0;
}

void sym_tree_formula_free(sym_formula_t *f) {
    if (!f) return;
    free(f->nodes);
    memset(f, 0, sizeof(*f));
}

int sym_tree_formula_copy(sym_formula_t *dst, const sym_formula_t *src) {
    if (!dst || !src || !src->nodes || src->n_nodes < 1) return -1;
    if (sym_tree_formula_init(dst, src->capacity > src->n_nodes ? src->capacity : src->n_nodes) != 0) {
        return -1;
    }
    memcpy(dst->nodes, src->nodes, (size_t)src->n_nodes * sizeof(sym_node_t));
    dst->n_nodes = src->n_nodes;
    dst->train_loss = src->train_loss;
    dst->valid_loss = src->valid_loss;
    dst->objective = src->objective;
    dst->complexity = src->complexity;
    return 0;
}

static int formula_assign(sym_formula_t *dst, const sym_formula_t *src) {
    sym_formula_t tmp;
    if (sym_tree_formula_copy(&tmp, src) != 0) return -1;
    sym_tree_formula_free(dst);
    *dst = tmp;
    return 0;
}

int sym_tree_formula_add_node(sym_formula_t *f, sym_node_t node) {
    if (!f || !f->nodes || f->n_nodes >= f->capacity) return -1;
    f->nodes[f->n_nodes] = node;
    return f->n_nodes++;
}

static double formula_complexity(const sym_formula_t *f) {
    if (!f || !f->nodes) return DBL_MAX;
    double c = 0.0;
    for (int32_t i = 0; i < f->n_nodes; i++) c += op_cost(f->nodes[i].op);
    return c;
}

int sym_tree_formula_validate(const sym_formula_t *f, int32_t n_features) {
    if (!f || !f->nodes || f->n_nodes < 1) {
        sym_set_error("formula has no nodes");
        return -1;
    }
    for (int32_t i = 0; i < f->n_nodes; i++) {
        const sym_node_t *n = &f->nodes[i];
        int32_t ar = op_arity(n->op);
        if (ar < 0) {
            sym_set_error("formula has invalid operator");
            return -1;
        }
        if (n->op == SYM_OP_VAR && (n->feature < 0 || n->feature >= n_features)) {
            sym_set_error("formula variable feature is out of bounds");
            return -1;
        }
        if (ar >= 1 && (n->left < 0 || n->left >= i)) {
            sym_set_error("formula left input violates topological order");
            return -1;
        }
        if (ar >= 2 && (n->right < 0 || n->right >= i)) {
            sym_set_error("formula right input violates topological order");
            return -1;
        }
    }
    return 0;
}

static int random_binary_op(sym_rng_t *rng, int operator_set) {
    int ops_basic[] = {SYM_OP_ADD, SYM_OP_SUB, SYM_OP_MUL, SYM_OP_DIV};
    int ops_full[] = {SYM_OP_ADD, SYM_OP_SUB, SYM_OP_MUL, SYM_OP_DIV, SYM_OP_MIN, SYM_OP_MAX};
    if (operator_set >= SYM_OPSET_FULL) return ops_full[rng_int(rng, 6)];
    return ops_basic[rng_int(rng, 4)];
}

static int random_unary_op(sym_rng_t *rng, int operator_set) {
    int ops_smooth[] = {SYM_OP_NEG, SYM_OP_SQRT, SYM_OP_LOG, SYM_OP_EXP, SYM_OP_SIN, SYM_OP_COS, SYM_OP_TANH};
    int ops_full[] = {SYM_OP_NEG, SYM_OP_ABS, SYM_OP_SQRT, SYM_OP_LOG, SYM_OP_EXP, SYM_OP_SIN, SYM_OP_COS, SYM_OP_TANH};
    if (operator_set >= SYM_OPSET_FULL) return ops_full[rng_int(rng, 8)];
    return ops_smooth[rng_int(rng, 7)];
}

int sym_tree_add_const_node(sym_formula_t *f, double value);
int sym_tree_add_op_node(sym_formula_t *f, int32_t op, int32_t left, int32_t right);

static int add_random_tree(
    sym_formula_t *f,
    sym_rng_t *rng,
    int32_t n_features,
    int32_t depth,
    const sym_params_t *params,
    int32_t budget
) {
    if (budget < 1 || f->n_nodes >= f->capacity) return -1;
    int remaining = f->capacity - f->n_nodes;
    if (budget > remaining) budget = remaining;
    int make_terminal = depth <= 0 || budget <= 1 || remaining <= 1 || rng_uniform(rng) < 0.22;
    if (make_terminal) {
        sym_node_t node;
        memset(&node, 0, sizeof(node));
        node.left = -1;
        node.right = -1;
        if (n_features > 0 && rng_uniform(rng) < 0.72) {
            node.op = SYM_OP_VAR;
            node.feature = rng_int(rng, n_features);
            node.value = 0.0;
        } else {
            node.op = SYM_OP_CONST;
            node.feature = -1;
            node.value = params->const_min + rng_uniform(rng) * (params->const_max - params->const_min);
        }
        return sym_tree_formula_add_node(f, node);
    }

    int allow_unary = params->operator_set >= SYM_OPSET_SMOOTH;
    int use_unary = allow_unary && (budget < 3 || rng_uniform(rng) < 0.35);
    if (use_unary) {
        int use_affine_input = budget >= 4 && rng_uniform(rng) < 0.45;
        int use_shift = use_affine_input && budget >= 6 && rng_uniform(rng) < 0.25;
        int affine_nodes = use_affine_input ? (use_shift ? 4 : 2) : 0;
        int child_budget = budget - 1 - affine_nodes;
        if (child_budget < 1) {
            use_affine_input = 0;
            use_shift = 0;
            affine_nodes = 0;
            child_budget = budget - 1;
        }

        int child = add_random_tree(f, rng, n_features, depth - 1, params, child_budget);
        if (child < 0) return -1;
        if (use_affine_input) {
            double scale = params->const_min + rng_uniform(rng) * (params->const_max - params->const_min);
            if (fabs(scale) < 0.1) scale = scale < 0.0 ? -0.1 : 0.1;
            int c_scale = sym_tree_add_const_node(f, scale);
            int mul = sym_tree_add_op_node(f, SYM_OP_MUL, child, c_scale);
            if (c_scale < 0 || mul < 0) return -1;
            child = mul;
            if (use_shift) {
                double shift = params->const_min + rng_uniform(rng) * (params->const_max - params->const_min);
                int c_shift = sym_tree_add_const_node(f, shift);
                int add = sym_tree_add_op_node(f, SYM_OP_ADD, child, c_shift);
                if (c_shift < 0 || add < 0) return -1;
                child = add;
            }
        }
        sym_node_t node;
        memset(&node, 0, sizeof(node));
        node.op = random_unary_op(rng, params->operator_set);
        node.left = child;
        node.right = -1;
        node.feature = -1;
        node.value = 0.0;
        return sym_tree_formula_add_node(f, node);
    }

    if (budget < 3) {
        sym_node_t node;
        memset(&node, 0, sizeof(node));
        node.left = -1;
        node.right = -1;
        node.op = SYM_OP_VAR;
        node.feature = rng_int(rng, n_features);
        return sym_tree_formula_add_node(f, node);
    }

    int left_budget = 1 + rng_int(rng, budget - 2);
    int right_budget = budget - 1 - left_budget;
    if (right_budget < 1) {
        right_budget = 1;
        left_budget = budget - 2;
    }

    int left = add_random_tree(f, rng, n_features, depth - 1, params, left_budget);
    if (left < 0) return -1;
    int right = add_random_tree(f, rng, n_features, depth - 1, params, right_budget);
    if (right < 0) return -1;
    sym_node_t node;
    memset(&node, 0, sizeof(node));
    node.op = random_binary_op(rng, params->operator_set);
    node.left = left;
    node.right = right;
    node.feature = -1;
    node.value = 0.0;
    return sym_tree_formula_add_node(f, node);
}

static int formula_random(sym_formula_t *f, sym_rng_t *rng, int32_t n_features, const sym_params_t *params) {
    if (sym_tree_formula_init(f, params->max_nodes) != 0) return -1;
    int root = add_random_tree(f, rng, n_features, params->max_depth, params, params->max_nodes);
    if (root < 0 || f->n_nodes < 1) {
        sym_tree_formula_free(f);
        sym_set_error("failed to create random formula");
        return -1;
    }
    f->complexity = formula_complexity(f);
    return 0;
}

static double logit_clamped(double p) {
    p = sym_tree_clamp_double(p, 1e-6, 1.0 - 1e-6);
    return log(p / (1.0 - p));
}

static void compute_seed_stats(
    const double *X,
    const double *target,
    const uint8_t *mask,
    int32_t nrow,
    int32_t ncol,
    const sym_params_t *params,
    double *slopes,
    double *intercepts,
    double *centers,
    double *constant
) {
    double sum_y = 0.0;
    int32_t n_train = 0;
    for (int32_t i = 0; i < nrow; i++) {
        if (mask && mask[i]) continue;
        sum_y += target[i];
        n_train++;
    }
    if (n_train <= 0) {
        for (int32_t i = 0; i < nrow; i++) sum_y += target[i];
        n_train = nrow;
    }

    double mean_y = n_train > 0 ? sum_y / (double)n_train : 0.0;
    double const_score = params->loss == SYM_LOSS_LOGLOSS ? logit_clamped(mean_y) : mean_y;
    if (!sym_tree_is_finite_value(const_score)) const_score = 0.0;
    *constant = const_score;

    for (int32_t j = 0; j < ncol; j++) {
        double sum_x = 0.0;
        double local_sum_y = 0.0;
        int32_t n = 0;
        for (int32_t i = 0; i < nrow; i++) {
            if (mask && mask[i]) continue;
            sum_x += X[(size_t)i * ncol + j];
            local_sum_y += target[i];
            n++;
        }
        if (n <= 0) {
            for (int32_t i = 0; i < nrow; i++) {
                sum_x += X[(size_t)i * ncol + j];
                local_sum_y += target[i];
            }
            n = nrow;
        }
        double mx = n > 0 ? sum_x / (double)n : 0.0;
        double my = n > 0 ? local_sum_y / (double)n : 0.0;
        centers[j] = sym_tree_is_finite_value(mx) ? mx : 0.0;
        double var = 0.0;
        double cov = 0.0;
        for (int32_t i = 0; i < nrow; i++) {
            if (mask && mask[i]) continue;
            double dx = X[(size_t)i * ncol + j] - mx;
            var += dx * dx;
            cov += dx * (target[i] - my);
        }
        if (var <= SYM_EPS) {
            slopes[j] = 0.0;
            intercepts[j] = const_score;
            continue;
        }
        double slope = cov / var;
        double intercept = my - slope * mx;
        if (params->loss == SYM_LOSS_LOGLOSS) {
            double scale = mean_y * (1.0 - mean_y);
            if (scale < 0.05) scale = 0.05;
            slope = sym_tree_clamp_double(slope / scale, -16.0, 16.0);
            intercept = const_score;
        }
        slopes[j] = sym_tree_is_finite_value(slope) ? slope : 0.0;
        intercepts[j] = sym_tree_is_finite_value(intercept) ? intercept : const_score;
    }
}

int sym_tree_add_const_node(sym_formula_t *f, double value) {
    sym_node_t node;
    memset(&node, 0, sizeof(node));
    node.op = SYM_OP_CONST;
    node.left = -1;
    node.right = -1;
    node.feature = -1;
    node.value = value;
    return sym_tree_formula_add_node(f, node);
}

static int add_var_node(sym_formula_t *f, int32_t feature) {
    sym_node_t node;
    memset(&node, 0, sizeof(node));
    node.op = SYM_OP_VAR;
    node.left = -1;
    node.right = -1;
    node.feature = feature;
    node.value = 0.0;
    return sym_tree_formula_add_node(f, node);
}

int sym_tree_add_op_node(sym_formula_t *f, int32_t op, int32_t left, int32_t right) {
    sym_node_t node;
    memset(&node, 0, sizeof(node));
    node.op = op;
    node.left = left;
    node.right = right;
    node.feature = -1;
    node.value = 0.0;
    return sym_tree_formula_add_node(f, node);
}

static int selected_contains(const int32_t *selected, int32_t n_selected, int32_t feature) {
    for (int32_t i = 0; i < n_selected; i++) {
        if (selected[i] == feature) return 1;
    }
    return 0;
}

static int32_t select_slope_rank(const double *slopes, int32_t n_features, const int32_t *selected, int32_t n_selected) {
    int32_t best = -1;
    double best_abs = -1.0;
    for (int32_t j = 0; j < n_features; j++) {
        if (selected_contains(selected, n_selected, j)) continue;
        double v = fabs(slopes[j]);
        if (v > best_abs) {
            best_abs = v;
            best = j;
        }
    }
    return best;
}

static int add_linear_combo_seed(
    sym_formula_t *f,
    int32_t n_features,
    const sym_params_t *params,
    double constant,
    const double *slopes
) {
    int32_t max_terms = (params->max_nodes - 1) / 4;
    if (max_terms < 1) return sym_tree_add_const_node(f, constant);
    if (max_terms > n_features) max_terms = n_features;
    if (max_terms > 63) max_terms = 63;

    int32_t root = sym_tree_add_const_node(f, constant);
    if (root < 0) return -1;
    int32_t selected[64];
    int32_t n_selected = 0;
    for (int32_t rank = 0; rank < max_terms; rank++) {
        int32_t feature = select_slope_rank(slopes, n_features, selected, n_selected);
        if (feature < 0 || fabs(slopes[feature]) <= SYM_EPS) break;
        selected[n_selected++] = feature;
        int c = sym_tree_add_const_node(f, slopes[feature]);
        int v = add_var_node(f, feature);
        int m = sym_tree_add_op_node(f, SYM_OP_MUL, c, v);
        int a = sym_tree_add_op_node(f, SYM_OP_ADD, root, m);
        if (c < 0 || v < 0 || m < 0 || a < 0) return -1;
        root = a;
    }
    return root;
}

int sym_tree_formula_seed(
    sym_formula_t *f,
    int32_t seed_index,
    int32_t n_features,
    const sym_params_t *params,
    double constant,
    const double *slopes,
    const double *intercepts,
    const double *centers
) {
    if (seed_index < 0 || n_features <= 0) return 0;
    if (sym_tree_formula_init(f, params->max_nodes) != 0) return -1;

    int ok = 0;
    if (seed_index == 0) {
        ok = sym_tree_add_const_node(f, constant) >= 0;
    } else if (seed_index == 1) {
        ok = sym_tree_add_const_node(f, 0.0) >= 0;
    } else if (seed_index == 2) {
        ok = sym_tree_add_const_node(f, 1.0) >= 0;
    } else {
        int32_t idx = seed_index - 3;
        if (idx < n_features) {
            ok = add_var_node(f, idx) >= 0;
        } else {
            idx -= n_features;
            if (idx < n_features) {
                if (params->max_nodes < 5) ok = 0;
                else {
                    int c0 = sym_tree_add_const_node(f, intercepts[idx]);
                    int c1 = sym_tree_add_const_node(f, slopes[idx]);
                    int v = add_var_node(f, idx);
                    int m = sym_tree_add_op_node(f, SYM_OP_MUL, c1, v);
                    ok = c0 >= 0 && c1 >= 0 && v >= 0 && m >= 0 &&
                        sym_tree_add_op_node(f, SYM_OP_ADD, c0, m) >= 0;
                }
            } else {
                idx -= n_features;
                if (idx < n_features) {
                    if (params->max_nodes < 4) ok = 0;
                    else {
                        int v = add_var_node(f, idx);
                        int c = sym_tree_add_const_node(f, centers[idx]);
                        int s = sym_tree_add_op_node(f, SYM_OP_SUB, v, c);
                        ok = v >= 0 && c >= 0 && s >= 0 && sym_tree_add_op_node(f, SYM_OP_MUL, s, s) >= 0;
                    }
                } else {
                    idx -= n_features;
                    if (idx == 0) {
                        ok = add_linear_combo_seed(f, n_features, params, constant, slopes) >= 0;
                    } else {
                        idx -= 1;
                        int64_t pair_count64 = (int64_t)n_features * (int64_t)n_features;
                        if (pair_count64 <= INT32_MAX / 4) {
                            if (params->operator_set >= SYM_OPSET_SMOOTH && (int64_t)idx < pair_count64) {
                                if (params->max_nodes >= 4) {
                                    int32_t pair = idx;
                                    int32_t a = pair / n_features;
                                    int32_t b = pair % n_features;
                                    int va = add_var_node(f, a);
                                    int vb = add_var_node(f, b);
                                    int m = sym_tree_add_op_node(f, SYM_OP_MUL, va, vb);
                                    ok = va >= 0 && vb >= 0 && m >= 0 && sym_tree_add_op_node(f, SYM_OP_SIN, m, -1) >= 0;
                                }
                            } else {
                                if (params->operator_set >= SYM_OPSET_SMOOTH) idx -= (int32_t)pair_count64;
                                if (idx >= 0 && (int64_t)idx < pair_count64 * 4 && params->max_nodes >= 3) {
                                    int32_t pair_count = (int32_t)pair_count64;
                                    int32_t mode = idx / pair_count;
                                    int32_t pair = idx % pair_count;
                                    int32_t a = pair / n_features;
                                    int32_t b = pair % n_features;
                                    int va = add_var_node(f, a);
                                    int vb = add_var_node(f, b);
                                    int op = SYM_OP_MUL;
                                    if (mode == 1) op = SYM_OP_ADD;
                                    else if (mode == 2) op = SYM_OP_SUB;
                                    else if (mode == 3) op = SYM_OP_DIV;
                                    ok = va >= 0 && vb >= 0 && sym_tree_add_op_node(f, op, va, vb) >= 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (!ok || f->n_nodes < 1 || sym_tree_formula_validate(f, n_features) != 0) {
        sym_tree_formula_free(f);
        return 0;
    }
    f->complexity = formula_complexity(f);
    return 1;
}

double sym_tree_formula_eval_row(const sym_formula_t *f, const double *row, int32_t n_features) {
    double stack[256];
    if (!f || !f->nodes || f->n_nodes < 1 || f->n_nodes > 256) return 0.0;
    for (int32_t i = 0; i < f->n_nodes; i++) {
        const sym_node_t *n = &f->nodes[i];
        double a = n->left >= 0 ? stack[n->left] : 0.0;
        double b = n->right >= 0 ? stack[n->right] : 0.0;
        double v = 0.0;
        switch (n->op) {
            case SYM_OP_CONST:
                v = n->value;
                break;
            case SYM_OP_VAR:
                v = (n->feature >= 0 && n->feature < n_features) ? row[n->feature] : 0.0;
                break;
            case SYM_OP_ADD:
                v = a + b;
                break;
            case SYM_OP_SUB:
                v = a - b;
                break;
            case SYM_OP_MUL:
                v = a * b;
                break;
            case SYM_OP_DIV: {
                double den = fabs(b) < SYM_EPS ? (b < 0.0 ? -SYM_EPS : SYM_EPS) : b;
                v = a / den;
                break;
            }
            case SYM_OP_NEG:
                v = -a;
                break;
            case SYM_OP_ABS:
                v = fabs(a);
                break;
            case SYM_OP_SQRT:
                v = sqrt(fabs(a));
                break;
            case SYM_OP_LOG:
                v = log(fabs(a) + SYM_EPS);
                break;
            case SYM_OP_EXP:
                v = exp(sym_tree_clamp_double(a, -40.0, 40.0));
                break;
            case SYM_OP_SIN:
                v = sin(a);
                break;
            case SYM_OP_COS:
                v = cos(a);
                break;
            case SYM_OP_TANH:
                v = tanh(a);
                break;
            case SYM_OP_MIN:
                v = a < b ? a : b;
                break;
            case SYM_OP_MAX:
                v = a > b ? a : b;
                break;
            default:
                v = 0.0;
                break;
        }
        if (!sym_tree_is_finite_value(v)) v = 0.0;
        stack[i] = sym_tree_clamp_double(v, -1e12, 1e12);
    }
    return stack[f->n_nodes - 1];
}

int sym_tree_collect_constant_nodes(const sym_formula_t *f, int32_t *nodes, int32_t max_nodes) {
    if (!f || !f->nodes || !nodes || max_nodes <= 0) return 0;
    int32_t n = 0;
    for (int32_t i = 0; i < f->n_nodes && n < max_nodes; i++) {
        if (f->nodes[i].op == SYM_OP_CONST) nodes[n++] = i;
    }
    return n;
}

double sym_tree_formula_eval_row_jac_constants(
    const sym_formula_t *f,
    const double *row,
    int32_t n_features,
    const int32_t *const_nodes,
    int32_t n_const,
    double *jac
) {
    double values[256];
    double adj[256];
    if (!f || !f->nodes || f->n_nodes < 1 || f->n_nodes > 256) {
        for (int32_t j = 0; j < n_const; j++) jac[j] = 0.0;
        return 0.0;
    }

    for (int32_t i = 0; i < f->n_nodes; i++) {
        const sym_node_t *n = &f->nodes[i];
        double a = n->left >= 0 ? values[n->left] : 0.0;
        double b = n->right >= 0 ? values[n->right] : 0.0;
        double v = 0.0;
        switch (n->op) {
            case SYM_OP_CONST:
                v = n->value;
                break;
            case SYM_OP_VAR:
                v = (n->feature >= 0 && n->feature < n_features) ? row[n->feature] : 0.0;
                break;
            case SYM_OP_ADD:
                v = a + b;
                break;
            case SYM_OP_SUB:
                v = a - b;
                break;
            case SYM_OP_MUL:
                v = a * b;
                break;
            case SYM_OP_DIV: {
                double den = fabs(b) < SYM_EPS ? (b < 0.0 ? -SYM_EPS : SYM_EPS) : b;
                v = a / den;
                break;
            }
            case SYM_OP_NEG:
                v = -a;
                break;
            case SYM_OP_ABS:
                v = fabs(a);
                break;
            case SYM_OP_SQRT:
                v = sqrt(fabs(a));
                break;
            case SYM_OP_LOG:
                v = log(fabs(a) + SYM_EPS);
                break;
            case SYM_OP_EXP:
                v = exp(sym_tree_clamp_double(a, -40.0, 40.0));
                break;
            case SYM_OP_SIN:
                v = sin(a);
                break;
            case SYM_OP_COS:
                v = cos(a);
                break;
            case SYM_OP_TANH:
                v = tanh(a);
                break;
            case SYM_OP_MIN:
                v = a < b ? a : b;
                break;
            case SYM_OP_MAX:
                v = a > b ? a : b;
                break;
            default:
                v = 0.0;
                break;
        }
        if (!sym_tree_is_finite_value(v)) v = 0.0;
        values[i] = sym_tree_clamp_double(v, -1e12, 1e12);
        adj[i] = 0.0;
    }

    adj[f->n_nodes - 1] = 1.0;
    for (int32_t i = f->n_nodes - 1; i >= 0; i--) {
        const sym_node_t *n = &f->nodes[i];
        double g = adj[i];
        if (!sym_tree_is_finite_value(g) || fabs(g) <= 0.0) continue;
        double a = n->left >= 0 ? values[n->left] : 0.0;
        double b = n->right >= 0 ? values[n->right] : 0.0;
        switch (n->op) {
            case SYM_OP_ADD:
                adj[n->left] += g;
                adj[n->right] += g;
                break;
            case SYM_OP_SUB:
                adj[n->left] += g;
                adj[n->right] -= g;
                break;
            case SYM_OP_MUL:
                adj[n->left] += g * b;
                adj[n->right] += g * a;
                break;
            case SYM_OP_DIV: {
                double den = fabs(b) < SYM_EPS ? (b < 0.0 ? -SYM_EPS : SYM_EPS) : b;
                adj[n->left] += g / den;
                if (fabs(b) >= SYM_EPS) adj[n->right] -= g * a / (den * den);
                break;
            }
            case SYM_OP_NEG:
                adj[n->left] -= g;
                break;
            case SYM_OP_ABS:
                if (a > 0.0) adj[n->left] += g;
                else if (a < 0.0) adj[n->left] -= g;
                break;
            case SYM_OP_SQRT:
                if (fabs(a) > SYM_EPS) adj[n->left] += g * 0.5 * (a < 0.0 ? -1.0 : 1.0) / sqrt(fabs(a));
                break;
            case SYM_OP_LOG:
                if (a > 0.0) adj[n->left] += g / (a + SYM_EPS);
                else if (a < 0.0) adj[n->left] -= g / (fabs(a) + SYM_EPS);
                break;
            case SYM_OP_EXP:
                if (a > -40.0 && a < 40.0) adj[n->left] += g * values[i];
                break;
            case SYM_OP_SIN:
                adj[n->left] += g * cos(a);
                break;
            case SYM_OP_COS:
                adj[n->left] -= g * sin(a);
                break;
            case SYM_OP_TANH: {
                double t = values[i];
                adj[n->left] += g * (1.0 - t * t);
                break;
            }
            case SYM_OP_MIN:
                if (a < b) adj[n->left] += g;
                else if (b < a) adj[n->right] += g;
                else {
                    adj[n->left] += 0.5 * g;
                    adj[n->right] += 0.5 * g;
                }
                break;
            case SYM_OP_MAX:
                if (a > b) adj[n->left] += g;
                else if (b > a) adj[n->right] += g;
                else {
                    adj[n->left] += 0.5 * g;
                    adj[n->right] += 0.5 * g;
                }
                break;
            default:
                break;
        }
    }

    for (int32_t j = 0; j < n_const; j++) {
        double v = adj[const_nodes[j]];
        jac[j] = sym_tree_is_finite_value(v) ? sym_tree_clamp_double(v, -1e12, 1e12) : 0.0;
    }
    return values[f->n_nodes - 1];
}

static double stable_logloss(double score, double target01) {
    score = sym_tree_clamp_double(score, -60.0, 60.0);
    if (score >= 0.0) {
        return (1.0 - target01) * score + log1p(exp(-score));
    }
    return -target01 * score + log1p(exp(score));
}

static double point_loss(double pred, double target, int loss, double huber_delta) {
    double e = pred - target;
    switch (loss) {
        case SYM_LOSS_MAE:
            return fabs(e);
        case SYM_LOSS_HUBER: {
            double ae = fabs(e);
            if (ae <= huber_delta) return 0.5 * e * e;
            return huber_delta * (ae - 0.5 * huber_delta);
        }
        case SYM_LOSS_LOGLOSS:
            return stable_logloss(pred, target);
        case SYM_LOSS_MSE:
        default:
            return e * e;
    }
}

int sym_tree_affine_fit_for_formula(
    const sym_formula_t *f,
    const double *X,
    const double *target,
    const uint8_t *is_valid,
    int32_t nrow,
    int32_t ncol,
    double *out_scale,
    double *out_bias
) {
    double sum_p = 0.0;
    double sum_y = 0.0;
    int32_t n = 0;
    for (int32_t i = 0; i < nrow; i++) {
        if (is_valid && is_valid[i]) continue;
        double p = sym_tree_formula_eval_row(f, X + (size_t)i * ncol, ncol);
        sum_p += p;
        sum_y += target[i];
        n++;
    }
    if (n <= 1) return 0;
    double mean_p = sum_p / (double)n;
    double mean_y = sum_y / (double)n;
    double var_p = 0.0;
    double cov = 0.0;
    for (int32_t i = 0; i < nrow; i++) {
        if (is_valid && is_valid[i]) continue;
        double p = sym_tree_formula_eval_row(f, X + (size_t)i * ncol, ncol);
        double dp = p - mean_p;
        var_p += dp * dp;
        cov += dp * (target[i] - mean_y);
    }
    if (var_p <= SYM_EPS) return 0;
    double scale = cov / var_p;
    double bias = mean_y - scale * mean_p;
    if (!sym_tree_is_finite_value(scale) || !sym_tree_is_finite_value(bias)) return 0;
    *out_scale = scale;
    *out_bias = bias;
    return 1;
}

void sym_tree_eval_formula_loss(
    sym_formula_t *f,
    const double *X,
    const double *target,
    const uint8_t *is_valid,
    int32_t nrow,
    int32_t ncol,
    const sym_params_t *params
) {
    double train = 0.0;
    double valid = 0.0;
    int32_t n_train = 0;
    int32_t n_valid = 0;
    for (int32_t i = 0; i < nrow; i++) {
        double pred = sym_tree_formula_eval_row(f, X + (size_t)i * ncol, ncol);
        double loss = point_loss(pred, target[i], params->loss, params->huber_delta);
        if (is_valid && is_valid[i]) {
            valid += loss;
            n_valid++;
        } else {
            train += loss;
            n_train++;
        }
    }
    f->train_loss = n_train > 0 ? train / (double)n_train : DBL_MAX;
    f->valid_loss = n_valid > 0 ? valid / (double)n_valid : f->train_loss;
    f->complexity = formula_complexity(f);
    f->objective = f->valid_loss + params->complexity_penalty * f->complexity;
    if (!sym_tree_is_finite_value(f->objective)) f->objective = DBL_MAX;
}

static void eval_loss_from_predictions(
    sym_formula_t *f,
    const double *pred,
    const double *target,
    const uint8_t *is_valid,
    int32_t nrow,
    const sym_params_t *params,
    double scale,
    double bias,
    double complexity
) {
    double train = 0.0;
    double valid = 0.0;
    int32_t n_train = 0;
    int32_t n_valid = 0;
    for (int32_t i = 0; i < nrow; i++) {
        double p = scale * pred[i] + bias;
        double loss = point_loss(p, target[i], params->loss, params->huber_delta);
        if (is_valid && is_valid[i]) {
            valid += loss;
            n_valid++;
        } else {
            train += loss;
            n_train++;
        }
    }
    f->train_loss = n_train > 0 ? train / (double)n_train : DBL_MAX;
    f->valid_loss = n_valid > 0 ? valid / (double)n_valid : f->train_loss;
    f->complexity = complexity;
    f->objective = f->valid_loss + params->complexity_penalty * f->complexity;
    if (!sym_tree_is_finite_value(f->objective)) f->objective = DBL_MAX;
}

static void eval_loss_from_predictions_rows(
    sym_formula_t *f,
    const double *pred,
    const double *target,
    const uint8_t *is_valid,
    const int32_t *rows,
    int32_t n_rows,
    const sym_params_t *params,
    double scale,
    double bias,
    double complexity
) {
    double train = 0.0;
    double valid = 0.0;
    int32_t n_train = 0;
    int32_t n_valid = 0;
    for (int32_t k = 0; k < n_rows; k++) {
        int32_t i = rows[k];
        double p = scale * pred[k] + bias;
        double loss = point_loss(p, target[i], params->loss, params->huber_delta);
        if (is_valid && is_valid[i]) {
            valid += loss;
            n_valid++;
        } else {
            train += loss;
            n_train++;
        }
    }
    f->train_loss = n_train > 0 ? train / (double)n_train : DBL_MAX;
    f->valid_loss = n_valid > 0 ? valid / (double)n_valid : f->train_loss;
    f->complexity = complexity;
    f->objective = f->valid_loss + params->complexity_penalty * f->complexity;
    if (!sym_tree_is_finite_value(f->objective)) f->objective = DBL_MAX;
}

static int affine_fit_from_predictions(
    const double *pred,
    const double *target,
    const uint8_t *is_valid,
    int32_t nrow,
    double *out_scale,
    double *out_bias
) {
    double sum_p = 0.0;
    double sum_y = 0.0;
    int32_t n = 0;
    for (int32_t i = 0; i < nrow; i++) {
        if (is_valid && is_valid[i]) continue;
        sum_p += pred[i];
        sum_y += target[i];
        n++;
    }
    if (n <= 1) return 0;
    double mean_p = sum_p / (double)n;
    double mean_y = sum_y / (double)n;
    double var_p = 0.0;
    double cov = 0.0;
    for (int32_t i = 0; i < nrow; i++) {
        if (is_valid && is_valid[i]) continue;
        double dp = pred[i] - mean_p;
        var_p += dp * dp;
        cov += dp * (target[i] - mean_y);
    }
    if (var_p <= SYM_EPS) return 0;
    double scale = cov / var_p;
    double bias = mean_y - scale * mean_p;
    if (!sym_tree_is_finite_value(scale) || !sym_tree_is_finite_value(bias)) return 0;
    *out_scale = scale;
    *out_bias = bias;
    return 1;
}

static int affine_fit_from_predictions_rows(
    const double *pred,
    const double *target,
    const uint8_t *is_valid,
    const int32_t *rows,
    int32_t n_rows,
    double *out_scale,
    double *out_bias
) {
    double sum_p = 0.0;
    double sum_y = 0.0;
    int32_t n = 0;
    for (int32_t k = 0; k < n_rows; k++) {
        int32_t i = rows[k];
        if (is_valid && is_valid[i]) continue;
        sum_p += pred[k];
        sum_y += target[i];
        n++;
    }
    if (n <= 1) return 0;
    double mean_p = sum_p / (double)n;
    double mean_y = sum_y / (double)n;
    double var_p = 0.0;
    double cov = 0.0;
    for (int32_t k = 0; k < n_rows; k++) {
        int32_t i = rows[k];
        if (is_valid && is_valid[i]) continue;
        double dp = pred[k] - mean_p;
        var_p += dp * dp;
        cov += dp * (target[i] - mean_y);
    }
    if (var_p <= SYM_EPS) return 0;
    double scale = cov / var_p;
    double bias = mean_y - scale * mean_p;
    if (!sym_tree_is_finite_value(scale) || !sym_tree_is_finite_value(bias)) return 0;
    *out_scale = scale;
    *out_bias = bias;
    return 1;
}

static void eval_formula_loss_search(
    sym_formula_t *f,
    const double *X,
    const double *target,
    const uint8_t *is_valid,
    int32_t nrow,
    int32_t ncol,
    const sym_params_t *params,
    double *pred_buf,
    double base_complexity
) {
    if (!pred_buf) {
        sym_tree_eval_formula_loss(f, X, target, is_valid, nrow, ncol, params);
        return;
    }

    for (int32_t i = 0; i < nrow; i++) {
        pred_buf[i] = sym_tree_formula_eval_row(f, X + (size_t)i * ncol, ncol);
    }

    if (params->loss != SYM_LOSS_MSE || f->n_nodes + 4 > f->capacity) {
        eval_loss_from_predictions(f, pred_buf, target, is_valid, nrow, params, 1.0, 0.0, base_complexity);
        return;
    }

    double scale = 1.0;
    double bias = 0.0;
    if (!affine_fit_from_predictions(pred_buf, target, is_valid, nrow, &scale, &bias) ||
        (fabs(scale - 1.0) < 1e-10 && fabs(bias) < 1e-10)) {
        eval_loss_from_predictions(f, pred_buf, target, is_valid, nrow, params, 1.0, 0.0, base_complexity);
        return;
    }

    double scaled_complexity = base_complexity + op_cost(SYM_OP_CONST) + op_cost(SYM_OP_MUL) +
        op_cost(SYM_OP_CONST) + op_cost(SYM_OP_ADD);
    eval_loss_from_predictions(f, pred_buf, target, is_valid, nrow, params, scale, bias, scaled_complexity);
}

static void eval_formula_loss_search_rows(
    sym_formula_t *f,
    const double *X,
    const double *target,
    const uint8_t *is_valid,
    int32_t nrow,
    int32_t ncol,
    const sym_params_t *params,
    const int32_t *rows,
    int32_t n_rows,
    double *pred_buf,
    double base_complexity
) {
    if (!rows || n_rows <= 0 || n_rows >= nrow) {
        eval_formula_loss_search(f, X, target, is_valid, nrow, ncol, params, pred_buf, base_complexity);
        return;
    }
    if (!pred_buf) {
        sym_tree_eval_formula_loss(f, X, target, is_valid, nrow, ncol, params);
        return;
    }

    for (int32_t k = 0; k < n_rows; k++) {
        int32_t i = rows[k];
        pred_buf[k] = sym_tree_formula_eval_row(f, X + (size_t)i * ncol, ncol);
    }

    if (params->loss != SYM_LOSS_MSE || f->n_nodes + 4 > f->capacity) {
        eval_loss_from_predictions_rows(f, pred_buf, target, is_valid, rows, n_rows, params, 1.0, 0.0, base_complexity);
        return;
    }

    double scale = 1.0;
    double bias = 0.0;
    if (!affine_fit_from_predictions_rows(pred_buf, target, is_valid, rows, n_rows, &scale, &bias) ||
        (fabs(scale - 1.0) < 1e-10 && fabs(bias) < 1e-10)) {
        eval_loss_from_predictions_rows(f, pred_buf, target, is_valid, rows, n_rows, params, 1.0, 0.0, base_complexity);
        return;
    }

    double scaled_complexity = base_complexity + op_cost(SYM_OP_CONST) + op_cost(SYM_OP_MUL) +
        op_cost(SYM_OP_CONST) + op_cost(SYM_OP_ADD);
    eval_loss_from_predictions_rows(f, pred_buf, target, is_valid, rows, n_rows, params, scale, bias, scaled_complexity);
}

static int cmp_formula_ptr(const void *a, const void *b) {
    const sym_formula_t *fa = *(const sym_formula_t * const *)a;
    const sym_formula_t *fb = *(const sym_formula_t * const *)b;
    if (fa->objective < fb->objective) return -1;
    if (fa->objective > fb->objective) return 1;
    if (fa->valid_loss < fb->valid_loss) return -1;
    if (fa->valid_loss > fb->valid_loss) return 1;
    if (fa->complexity < fb->complexity) return -1;
    if (fa->complexity > fb->complexity) return 1;
    return fa->n_nodes - fb->n_nodes;
}

int sym_tree_formulas_equal_nodes(const sym_formula_t *a, const sym_formula_t *b) {
    if (!a || !b || a->n_nodes != b->n_nodes) return 0;
    return memcmp(a->nodes, b->nodes, (size_t)a->n_nodes * sizeof(sym_node_t)) == 0;
}

static int copy_subtree(
    const sym_formula_t *src,
    int32_t idx,
    sym_formula_t *dst,
    int32_t *map
) {
    if (idx < 0 || idx >= src->n_nodes) return -1;
    if (map[idx] >= 0) return map[idx];
    sym_node_t n = src->nodes[idx];
    int ar = op_arity(n.op);
    if (ar >= 1) {
        int left = copy_subtree(src, n.left, dst, map);
        if (left < 0) return -1;
        n.left = left;
    }
    if (ar >= 2) {
        int right = copy_subtree(src, n.right, dst, map);
        if (right < 0) return -1;
        n.right = right;
    }
    int out = sym_tree_formula_add_node(dst, n);
    if (out < 0) return -1;
    map[idx] = out;
    return out;
}

static int copy_tree_replacing(
    const sym_formula_t *a,
    int32_t idx,
    int32_t cut_a,
    const sym_formula_t *b,
    int32_t cut_b,
    sym_formula_t *dst,
    int32_t *map_a,
    int32_t *map_b
) {
    if (idx == cut_a) return copy_subtree(b, cut_b, dst, map_b);
    if (map_a[idx] >= 0) return map_a[idx];
    sym_node_t n = a->nodes[idx];
    int ar = op_arity(n.op);
    if (ar >= 1) {
        int left = copy_tree_replacing(a, n.left, cut_a, b, cut_b, dst, map_a, map_b);
        if (left < 0) return -1;
        n.left = left;
    }
    if (ar >= 2) {
        int right = copy_tree_replacing(a, n.right, cut_a, b, cut_b, dst, map_a, map_b);
        if (right < 0) return -1;
        n.right = right;
    }
    int out = sym_tree_formula_add_node(dst, n);
    if (out < 0) return -1;
    map_a[idx] = out;
    return out;
}

static int formula_crossover(
    sym_formula_t *out,
    const sym_formula_t *a,
    const sym_formula_t *b,
    sym_rng_t *rng,
    const sym_params_t *params
) {
    if (sym_tree_formula_init(out, params->max_nodes) != 0) return -1;
    int32_t cut_a = rng_int(rng, a->n_nodes);
    int32_t cut_b = rng_int(rng, b->n_nodes);
    int32_t map_a[256];
    int32_t map_b[256];
    for (int i = 0; i < 256; i++) {
        map_a[i] = -1;
        map_b[i] = -1;
    }
    int root = copy_tree_replacing(a, a->n_nodes - 1, cut_a, b, cut_b, out, map_a, map_b);
    if (root < 0 || out->n_nodes < 1) {
        sym_tree_formula_free(out);
        return sym_tree_formula_copy(out, a);
    }
    out->complexity = formula_complexity(out);
    return 0;
}

static int formula_mutate(
    sym_formula_t *out,
    const sym_formula_t *in,
    sym_rng_t *rng,
    int32_t n_features,
    const sym_params_t *params
) {
    if (rng_uniform(rng) < 0.5) {
        if (sym_tree_formula_init(out, params->max_nodes) != 0) return -1;
        int32_t cut = rng_int(rng, in->n_nodes);
        int32_t map_a[256];
        int32_t map_b[256];
        for (int i = 0; i < 256; i++) {
            map_a[i] = -1;
            map_b[i] = -1;
        }
        sym_formula_t subtree;
        if (formula_random(&subtree, rng, n_features, params) != 0) {
            sym_tree_formula_free(out);
            return -1;
        }
        int root = copy_tree_replacing(in, in->n_nodes - 1, cut, &subtree, subtree.n_nodes - 1, out, map_a, map_b);
        sym_tree_formula_free(&subtree);
        if (root < 0 || out->n_nodes < 1) {
            sym_tree_formula_free(out);
            return sym_tree_formula_copy(out, in);
        }
    } else {
        if (sym_tree_formula_copy(out, in) != 0) return -1;
        int32_t i = rng_int(rng, out->n_nodes);
        sym_node_t *n = &out->nodes[i];
        if (n->op == SYM_OP_CONST || rng_uniform(rng) < params->constant_rate) {
            n->op = SYM_OP_CONST;
            n->left = -1;
            n->right = -1;
            n->feature = -1;
            n->value += rng_normalish(rng) * 0.5 * (params->const_max - params->const_min);
            n->value = sym_tree_clamp_double(n->value, params->const_min * 4.0, params->const_max * 4.0);
        } else if (n->op == SYM_OP_VAR) {
            n->feature = rng_int(rng, n_features);
        } else {
            int ar = op_arity(n->op);
            n->op = ar == 1 ? random_unary_op(rng, params->operator_set) : random_binary_op(rng, params->operator_set);
        }
    }
    out->complexity = formula_complexity(out);
    return 0;
}

static int tournament_select_slice(
    const sym_formula_t *pop,
    int32_t start,
    int32_t end,
    int32_t tournament,
    sym_rng_t *rng
) {
    if (end <= start + 1) return start;
    int32_t n = tournament;
    if (n < 1) n = 1;
    if (n > 128) n = 128;
    int32_t best = start + rng_int(rng, end - start);
    for (int32_t i = 1; i < n; i++) {
        int32_t cand = start + rng_int(rng, end - start);
        if (pop[cand].objective < pop[best].objective) best = cand;
    }
    return best;
}

static int current_warmup_max_nodes(const sym_params_t *params, int32_t gen) {
    if (!params || params->warmup_generations <= 0 || params->warmup_min_nodes >= params->max_nodes) {
        return params ? params->max_nodes : 1;
    }
    if (gen >= params->warmup_generations) return params->max_nodes;
    double t = (double)(gen + 1) / (double)params->warmup_generations;
    int32_t span = params->max_nodes - params->warmup_min_nodes;
    int32_t nodes = params->warmup_min_nodes + (int32_t)floor((double)span * t + 0.5);
    if (nodes < params->warmup_min_nodes) nodes = params->warmup_min_nodes;
    if (nodes > params->max_nodes) nodes = params->max_nodes;
    if ((nodes & 1) == 0) nodes--;
    if (nodes < 1) nodes = 1;
    return nodes;
}

static int build_row_sample(
    int32_t *rows,
    int32_t nrow,
    int32_t sample_size,
    sym_rng_t *rng
) {
    if (!rows || sample_size <= 0 || sample_size >= nrow) return 0;
    for (int32_t i = 0; i < nrow; i++) rows[i] = i;
    for (int32_t i = 0; i < sample_size; i++) {
        int32_t j = i + rng_int(rng, nrow - i);
        int32_t t = rows[i];
        rows[i] = rows[j];
        rows[j] = t;
    }
    return sample_size;
}

static int copy_best_from_range(
    sym_formula_t *dst,
    const sym_formula_t *pop,
    int32_t start,
    int32_t end,
    int32_t rank
) {
    if (rank < 0) return -1;
    int32_t selected[64];
    int32_t n_selected = 0;
    if (rank >= 64) rank = 63;
    for (int32_t r = 0; r <= rank; r++) {
        int32_t best_i = -1;
        for (int32_t i = start; i < end; i++) {
            if (!pop[i].nodes) continue;
            int used = 0;
            for (int32_t s = 0; s < n_selected; s++) {
                if (selected[s] == i) {
                    used = 1;
                    break;
                }
            }
            if (used) continue;
            if (best_i < 0 ||
                pop[i].objective < pop[best_i].objective ||
                (pop[i].objective == pop[best_i].objective && pop[i].complexity < pop[best_i].complexity)) {
                best_i = i;
            }
        }
        if (best_i < 0) return -1;
        selected[n_selected++] = best_i;
    }
    return sym_tree_formula_copy(dst, &pop[selected[n_selected - 1]]);
}

static int make_offspring(
    sym_formula_t *out,
    const sym_formula_t *pop,
    int32_t start,
    int32_t end,
    sym_rng_t *rng,
    int32_t ncol,
    const sym_params_t *params
) {
    int a_idx = tournament_select_slice(pop, start, end, params->tournament_size, rng);
    const sym_formula_t *parent_a = &pop[a_idx];
    if (rng_uniform(rng) < params->crossover_rate && end > start + 1) {
        int b_idx = tournament_select_slice(pop, start, end, params->tournament_size, rng);
        if (formula_crossover(out, parent_a, &pop[b_idx], rng, params) != 0) return -1;
    } else {
        if (sym_tree_formula_copy(out, parent_a) != 0) return -1;
    }
    if (rng_uniform(rng) < params->mutation_rate) {
        sym_formula_t mutated;
        if (formula_mutate(&mutated, out, rng, ncol, params) != 0) return -1;
        sym_tree_formula_free(out);
        *out = mutated;
    }
    return 0;
}

static int build_validation_mask(uint8_t *mask, int32_t nrow, double fraction, sym_rng_t *rng) {
    memset(mask, 0, (size_t)nrow);
    int32_t n_valid = (int32_t)floor((double)nrow * fraction);
    if (n_valid <= 0 || n_valid >= nrow) return 0;
    int32_t *idx = (int32_t *)malloc((size_t)nrow * sizeof(int32_t));
    if (!idx) return -1;
    for (int32_t i = 0; i < nrow; i++) idx[i] = i;
    for (int32_t i = nrow - 1; i > 0; i--) {
        int32_t j = rng_int(rng, i + 1);
        int32_t t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
    for (int32_t i = 0; i < n_valid; i++) mask[idx[i]] = 1;
    free(idx);
    return n_valid;
}

static int formula_dominates(const sym_formula_t *a, const sym_formula_t *b) {
    const double eps = 1e-12;
    int loss_ok = a->valid_loss <= b->valid_loss + eps;
    int complexity_ok = a->complexity <= b->complexity + eps;
    int strict = a->valid_loss + eps < b->valid_loss || a->complexity + eps < b->complexity;
    return loss_ok && complexity_ok && strict;
}

int sym_tree_add_to_frontier(sym_formula_t *frontier, int32_t *n_frontier, int32_t cap, const sym_formula_t *cand) {
    for (int32_t i = 0; i < *n_frontier; i++) {
        if (sym_tree_formulas_equal_nodes(&frontier[i], cand)) {
            if (cand->objective < frontier[i].objective) {
                formula_assign(&frontier[i], cand);
            }
            return 0;
        }
    }

    int32_t dominated_replace = -1;
    double dominated_worst = -DBL_MAX;
    for (int32_t i = 0; i < *n_frontier; i++) {
        if (*n_frontier >= cap && formula_dominates(&frontier[i], cand)) return 0;
        if (formula_dominates(cand, &frontier[i]) && frontier[i].objective > dominated_worst) {
            dominated_worst = frontier[i].objective;
            dominated_replace = i;
        }
    }

    int insert = -1;
    if (dominated_replace >= 0) {
        insert = dominated_replace;
    } else if (*n_frontier < cap) {
        insert = *n_frontier;
        (*n_frontier)++;
    } else {
        double worst = -DBL_MAX;
        int32_t worst_i = -1;
        for (int32_t i = 0; i < *n_frontier; i++) {
            double v = frontier[i].objective;
            if (v > worst) {
                worst = v;
                worst_i = i;
            }
        }
        if (cand->objective >= worst) return 0;
        insert = worst_i;
    }
    return formula_assign(&frontier[insert], cand);
}

static void sort_frontier(sym_formula_t *frontier, int32_t n_frontier) {
    for (int32_t i = 0; i < n_frontier; i++) {
        for (int32_t j = i + 1; j < n_frontier; j++) {
            if (frontier[j].objective < frontier[i].objective ||
                (frontier[j].objective == frontier[i].objective && frontier[j].complexity < frontier[i].complexity)) {
                sym_formula_t t = frontier[i];
                frontier[i] = frontier[j];
                frontier[j] = t;
            }
        }
    }
}

static int formula_exists_in_slots(const sym_formula_t *items, int32_t n_items, const sym_formula_t *cand);

static int complexity_bucket_for(const sym_formula_t *f, const sym_params_t *params) {
    if (!f || !params || params->complexity_bucket_width <= 0.0) return 0;
    double c = f->complexity;
    if (!sym_tree_is_finite_value(c) || c < 0.0) c = formula_complexity(f);
    int32_t bucket = (int32_t)floor(c / params->complexity_bucket_width);
    return bucket < 0 ? 0 : bucket;
}

static int add_to_complexity_hof(
    sym_formula_t *hof,
    int32_t *n_hof,
    int32_t cap,
    const sym_formula_t *cand,
    const sym_params_t *params
) {
    if (!hof || !n_hof || cap <= 0 || !cand || !cand->nodes || cand->n_nodes <= 0) return 0;
    if (!sym_tree_is_finite_value(cand->valid_loss) || cand->valid_loss == DBL_MAX) return 0;
    int32_t bucket = complexity_bucket_for(cand, params);
    for (int32_t i = 0; i < *n_hof; i++) {
        if (complexity_bucket_for(&hof[i], params) == bucket) {
            if (cand->valid_loss < hof[i].valid_loss ||
                (cand->valid_loss == hof[i].valid_loss && cand->complexity < hof[i].complexity)) {
                return formula_assign(&hof[i], cand);
            }
            return 0;
        }
    }
    if (*n_hof < cap) {
        int32_t slot = *n_hof;
        (*n_hof)++;
        return formula_assign(&hof[slot], cand);
    }

    int32_t worst_i = -1;
    double worst_loss = -DBL_MAX;
    for (int32_t i = 0; i < *n_hof; i++) {
        double v = hof[i].valid_loss;
        if (v > worst_loss) {
            worst_loss = v;
            worst_i = i;
        }
    }
    if (worst_i < 0 || cand->valid_loss >= worst_loss) return 0;
    return formula_assign(&hof[worst_i], cand);
}

static double selector_score(
    const sym_formula_t *cand,
    const sym_formula_t *items,
    int32_t n_items
) {
    if (!cand || !items || n_items <= 0 || cand->valid_loss <= 0.0 || !sym_tree_is_finite_value(cand->valid_loss)) {
        return -DBL_MAX;
    }
    double prev_loss = DBL_MAX;
    double prev_complexity = 0.0;
    for (int32_t i = 0; i < n_items; i++) {
        const sym_formula_t *other = &items[i];
        if (!other->nodes || other == cand) continue;
        if (!sym_tree_is_finite_value(other->valid_loss) || other->valid_loss <= 0.0) continue;
        if (other->complexity + 1e-12 < cand->complexity && other->valid_loss < prev_loss) {
            prev_loss = other->valid_loss;
            prev_complexity = other->complexity;
        }
    }
    if (prev_loss == DBL_MAX || prev_loss <= cand->valid_loss) return -DBL_MAX;
    double dc = cand->complexity - prev_complexity;
    if (dc <= 1e-12) dc = 1.0;
    double score = (log(prev_loss + 1e-300) - log(cand->valid_loss + 1e-300)) / dc;
    return sym_tree_is_finite_value(score) ? score : -DBL_MAX;
}

static int formula_better_for_final(
    const sym_formula_t *a,
    const sym_formula_t *b,
    const sym_formula_t *pool,
    int32_t n_pool,
    const sym_params_t *params
) {
    if (!a || !a->nodes) return 0;
    if (!b || !b->nodes) return 1;
    int selector = params ? params->final_selector : SYM_FINAL_OBJECTIVE;
    if (selector == SYM_FINAL_LOSS) {
        if (a->valid_loss < b->valid_loss) return 1;
        if (a->valid_loss > b->valid_loss) return 0;
        if (a->complexity < b->complexity) return 1;
        if (a->complexity > b->complexity) return 0;
        return a->objective < b->objective;
    }
    if (selector == SYM_FINAL_SCORE) {
        double min_loss = DBL_MAX;
        for (int32_t i = 0; i < n_pool; i++) {
            if (pool[i].nodes && sym_tree_is_finite_value(pool[i].valid_loss) && pool[i].valid_loss < min_loss) {
                min_loss = pool[i].valid_loss;
            }
        }
        double loss_gate = sym_tree_is_finite_value(min_loss) && min_loss < DBL_MAX ? min_loss * 1.5 + 1e-12 : DBL_MAX;
        int a_ok = a->valid_loss <= loss_gate;
        int b_ok = b->valid_loss <= loss_gate;
        if (a_ok && !b_ok) return 1;
        if (!a_ok && b_ok) return 0;
        double sa = selector_score(a, pool, n_pool);
        double sb = selector_score(b, pool, n_pool);
        if (sa > sb + 1e-12) return 1;
        if (sb > sa + 1e-12) return 0;
        if (a->valid_loss < b->valid_loss) return 1;
        if (a->valid_loss > b->valid_loss) return 0;
        if (a->complexity < b->complexity) return 1;
        if (a->complexity > b->complexity) return 0;
        return a->objective < b->objective;
    }
    if (a->objective < b->objective) return 1;
    if (a->objective > b->objective) return 0;
    if (a->valid_loss < b->valid_loss) return 1;
    if (a->valid_loss > b->valid_loss) return 0;
    return a->complexity < b->complexity;
}

static int select_final_formula(
    sym_formula_t *out,
    sym_formula_t *frontier,
    int32_t n_frontier,
    sym_formula_t *hof,
    int32_t n_hof,
    const sym_params_t *params
) {
    int32_t n_pool = n_frontier + n_hof;
    if (n_pool <= 0) return -1;
    sym_formula_t *pool = (sym_formula_t *)calloc((size_t)n_pool, sizeof(sym_formula_t));
    if (!pool) return -1;
    int32_t n = 0;
    for (int32_t i = 0; i < n_frontier; i++) {
        if (frontier[i].nodes && sym_tree_formula_copy(&pool[n++], &frontier[i]) != 0) goto fail;
    }
    for (int32_t i = 0; i < n_hof; i++) {
        if (hof[i].nodes && sym_tree_formula_copy(&pool[n++], &hof[i]) != 0) goto fail;
    }
    if (n <= 0) goto fail;
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) {
        if (formula_better_for_final(&pool[i], &pool[best], pool, n, params)) best = i;
    }
    int rc = sym_tree_formula_copy(out, &pool[best]);
    if (frontier && n_frontier > 0 && !formula_exists_in_slots(frontier, n_frontier, &pool[best])) {
        int32_t worst = 0;
        for (int32_t i = 1; i < n_frontier; i++) {
            if (frontier[i].objective > frontier[worst].objective) worst = i;
        }
        formula_assign(&frontier[worst], &pool[best]);
    }
    for (int32_t i = 0; i < n; i++) sym_tree_formula_free(&pool[i]);
    free(pool);
    return rc;

fail:
    for (int32_t i = 0; i < n_pool; i++) sym_tree_formula_free(&pool[i]);
    free(pool);
    return -1;
}

static int formula_exists_in_slots(const sym_formula_t *items, int32_t n_items, const sym_formula_t *cand) {
    for (int32_t i = 0; i < n_items; i++) {
        if (items[i].nodes && sym_tree_formulas_equal_nodes(&items[i], cand)) return 1;
    }
    return 0;
}

/* Search is a pull state machine: a pending batch is immutable until accepted.
 * Local coefficient refinement remains the canonical C algorithm; its dependent
 * trial evaluations must not be reordered with evolutionary scoring batches. */
enum search_phase {
    SEARCH_SEED, SEARCH_GENERATION, SEARCH_RESAMPLE, SEARCH_SORT,
    SEARCH_FRONTIER, SEARCH_TOP, SEARCH_AFTER_TOP, SEARCH_ISLAND,
    SEARCH_CHILDREN, SEARCH_REFINE, SEARCH_FINAL, SEARCH_FINISH_TREE,
    SEARCH_DONE, SEARCH_FAILED, SEARCH_TAKEN
};

struct sym_search {
    sym_model_t *model;
    sym_params_t params, active;
    double *X, *y, *target;
    uint8_t *mask;
    int32_t nrow, ncol, cls, capacity, cursor, gen, stale;
    int32_t island, start, end, out_i, brood, n_hof, n_score_rows;
    enum search_phase phase;
    sym_rng_t rng;
    double best_objective, seed_constant;
    double *seed_slopes, *seed_intercepts, *seed_centers, *pred_buf, *base_complexities;
    int32_t *score_rows, *slots;
    sym_formula_t *pop, *next, *hof, *scratch;
    sym_formula_t **sorted;
    const sym_formula_t **views;
    sym_batch_t batch;
    uint32_t serial;
    int pending;
};

static void search_clear_population(sym_search_t *s) {
    for (int32_t i = 0; i < s->params.population; i++) {
        if (s->pop) sym_tree_formula_free(&s->pop[i]);
        if (s->next) sym_tree_formula_free(&s->next[i]);
    }
    for (int32_t i = 0; i < s->n_hof; i++) sym_tree_formula_free(&s->hof[i]);
    s->n_hof = 0;
    for (int32_t i = 0; i < s->capacity; i++) {
        if (s->scratch) sym_tree_formula_free(&s->scratch[i]);
    }
}

void sym_search_free(sym_search_t *s) {
    if (!s) return;
    search_clear_population(s);
    sym_free(s->model);
    free(s->X); free(s->y); free(s->target); free(s->mask);
    free(s->pop); free(s->next); free(s->hof); free(s->scratch);
    free(s->sorted); free(s->views); free(s->slots);
    free(s->seed_slopes); free(s->seed_intercepts); free(s->seed_centers);
    free(s->pred_buf); free(s->score_rows); free(s->base_complexities);
    free(s);
}

static void search_start_tree(sym_search_t *s) {
    search_clear_population(s);
    uint32_t seed = s->params.seed + 7919u + (uint32_t)s->cls * 104729u;
    s->rng.state = seed ? seed : 1u;
    for (int32_t i = 0; i < s->nrow; i++) {
        s->target[i] = s->params.task != SYM_TASK_CLASSIFICATION ? s->y[i]
            : s->params.n_classes <= 2 ? (s->y[i] > 0.0 ? 1.0 : 0.0)
            : ((int32_t)llround(s->y[i]) == s->cls ? 1.0 : 0.0);
    }
    compute_seed_stats(s->X, s->target, s->mask, s->nrow, s->ncol, &s->params,
        s->seed_slopes, s->seed_intercepts, s->seed_centers, &s->seed_constant);
    s->active = s->params;
    s->active.max_nodes = current_warmup_max_nodes(&s->params, 0);
    if (s->active.max_depth > s->active.max_nodes) s->active.max_depth = s->active.max_nodes;
    s->n_score_rows = build_row_sample(s->score_rows, s->nrow, s->params.row_sample_size, &s->rng);
    s->best_objective = DBL_MAX;
    s->cursor = s->gen = s->stale = 0;
    s->phase = SEARCH_SEED;
}

sym_search_t *sym_search_new(const double *X, int32_t nrow, int32_t ncol,
                            const double *y, const sym_params_t *params_in,
                            int32_t batch_capacity) {
    if (!X || !y || nrow <= 0 || ncol <= 0 || batch_capacity < 1 || batch_capacity > 65536 ||
        (size_t)nrow > SIZE_MAX / sizeof(double) / (size_t)ncol) {
        sym_set_error("invalid search input shape or batch capacity (1..65536)");
        return NULL;
    }
    sym_params_t p;
    sym_params_init(&p);
    if (params_in) p = *params_in;
    normalize_params(&p, nrow, ncol);
    if ((p.task != SYM_TASK_REGRESSION && p.task != SYM_TASK_CLASSIFICATION && p.task != SYM_TASK_TRANSFORMER) ||
        (p.task == SYM_TASK_CLASSIFICATION && p.n_classes < 2)) {
        sym_set_error("invalid search task or class count");
        return NULL;
    }
    sym_search_t *s = (sym_search_t *)calloc(1, sizeof(*s));
    if (!s) { sym_set_error("out of memory allocating search"); return NULL; }
    s->params = p; s->nrow = nrow; s->ncol = ncol; s->capacity = batch_capacity;
    s->model = (sym_model_t *)calloc(1, sizeof(sym_model_t));
    if (!s->model) goto oom;
    sym_model_t *m = s->model;
    m->task = p.task; m->n_features = ncol;
    m->n_classes = p.task == SYM_TASK_CLASSIFICATION ? p.n_classes : 0;
    m->seed = p.seed; m->params = p;
    m->n_outputs = p.task == SYM_TASK_TRANSFORMER ? p.top_k
        : p.task == SYM_TASK_CLASSIFICATION && p.n_classes > 2 ? p.n_classes : 1;
    m->n_formulas = m->n_outputs;
    m->formulas = (sym_formula_t *)calloc((size_t)m->n_formulas, sizeof(sym_formula_t));
    m->frontier = (sym_formula_t *)calloc((size_t)p.frontier_size, sizeof(sym_formula_t));
    s->X = (double *)malloc((size_t)nrow * ncol * sizeof(double));
    s->y = (double *)malloc((size_t)nrow * sizeof(double));
    s->target = (double *)malloc((size_t)nrow * sizeof(double));
    s->mask = (uint8_t *)calloc((size_t)nrow, 1);
    s->pop = (sym_formula_t *)calloc((size_t)p.population, sizeof(sym_formula_t));
    s->next = (sym_formula_t *)calloc((size_t)p.population, sizeof(sym_formula_t));
    s->sorted = (sym_formula_t **)calloc((size_t)p.population, sizeof(sym_formula_t *));
    s->scratch = (sym_formula_t *)calloc((size_t)batch_capacity, sizeof(sym_formula_t));
    s->views = (const sym_formula_t **)calloc((size_t)batch_capacity, sizeof(sym_formula_t *));
    s->slots = (int32_t *)calloc((size_t)batch_capacity, sizeof(int32_t));
    s->base_complexities = (double *)malloc((size_t)batch_capacity * sizeof(double));
    s->seed_slopes = (double *)calloc((size_t)ncol, sizeof(double));
    s->seed_intercepts = (double *)calloc((size_t)ncol, sizeof(double));
    s->seed_centers = (double *)calloc((size_t)ncol, sizeof(double));
    s->pred_buf = (double *)malloc((size_t)nrow * sizeof(double));
    if (p.row_sample_size > 0) s->score_rows = (int32_t *)malloc((size_t)nrow * sizeof(int32_t));
    if (p.complexity_hof_size > 0) s->hof = (sym_formula_t *)calloc((size_t)p.complexity_hof_size, sizeof(sym_formula_t));
    if (!m->formulas || !m->frontier || !s->X || !s->y || !s->target || !s->mask ||
        !s->pop || !s->next || !s->sorted || !s->scratch || !s->views || !s->slots || !s->base_complexities ||
        !s->seed_slopes || !s->seed_intercepts || !s->seed_centers || !s->pred_buf ||
        (p.row_sample_size > 0 && !s->score_rows) || (p.complexity_hof_size > 0 && !s->hof)) goto oom;
    memcpy(s->X, X, (size_t)nrow * ncol * sizeof(double));
    memcpy(s->y, y, (size_t)nrow * sizeof(double));
    sym_rng_t split_rng = { p.seed ? p.seed : 1u };
    if (build_validation_mask(s->mask, nrow, p.validation_fraction, &split_rng) < 0) goto oom;
    if (p.task == SYM_TASK_CLASSIFICATION) s->params.loss = SYM_LOSS_LOGLOSS;
    search_start_tree(s);
    return s;
oom:
    sym_set_error("out of memory allocating search buffers");
    sym_search_free(s);
    return NULL;
}

static int search_issue(sym_search_t *s, int32_t count, int sampled, int stage,
                        const sym_batch_t **out) {
    if (s->serial == UINT32_MAX) { sym_set_error("search batch sequence exhausted"); s->phase = SEARCH_FAILED; return -1; }
    /* Complexity depends only on immutable descriptors, not host scores. Keep
     * it once per proposal for the scorer and objective construction. */
    for (int32_t i = 0; i < count; i++) s->base_complexities[i] = formula_complexity(s->views[i]);
    s->batch.id = ++s->serial;
    s->batch.stage = stage;
    s->batch.count = count;
    s->batch.formulas = s->views;
    s->batch.X = s->X; s->batch.target = s->target; s->batch.validation_mask = s->mask;
    s->batch.nrow = s->nrow; s->batch.ncol = s->ncol;
    s->batch.row_indices = sampled && s->n_score_rows > 0 ? s->score_rows : NULL;
    s->batch.rows = s->batch.row_indices ? s->n_score_rows : s->nrow;
    s->pending = 1;
    *out = &s->batch;
    return 1;
}

static void search_archive(sym_search_t *s, const sym_formula_t *f) {
    sym_tree_add_to_frontier(s->model->frontier, &s->model->n_frontier, s->params.frontier_size, f);
    add_to_complexity_hof(s->hof, &s->n_hof, s->params.complexity_hof_size, f, &s->params);
}

static int search_finish_tree(sym_search_t *s) {
    sym_model_t *m = s->model;
    for (int32_t i = 0; i < s->params.population; i++) s->sorted[i] = &s->pop[i];
    qsort(s->sorted, (size_t)s->params.population, sizeof(sym_formula_t *), cmp_formula_ptr);
    for (int32_t i = 0; i < s->params.frontier_size && i < s->params.population; i++) search_archive(s, s->sorted[i]);
    sort_frontier(m->frontier, m->n_frontier);
    if (m->n_frontier <= 0) { sym_set_error("symbolic search produced no frontier"); return -1; }
    for (int32_t i = 0; i < m->n_frontier; i++) {
        sym_tree_try_affine_rescale(&m->frontier[i], s->X, s->target, s->mask, s->nrow, s->ncol, &s->params);
        add_to_complexity_hof(s->hof, &s->n_hof, s->params.complexity_hof_size, &m->frontier[i], &s->params);
    }
    sort_frontier(m->frontier, m->n_frontier);
    int32_t refine_n = m->n_frontier < s->params.top_k ? m->n_frontier : s->params.top_k;
    if (refine_n > 12) refine_n = 12;
    for (int32_t i = 0; i < refine_n; i++) {
        sym_tree_improve_constants(&m->frontier[i], s->X, s->target, s->mask, s->nrow, s->ncol, &s->params);
        add_to_complexity_hof(s->hof, &s->n_hof, s->params.complexity_hof_size, &m->frontier[i], &s->params);
    }
    if (sym_tree_try_linear_basis_combo(m->frontier, &m->n_frontier, s->params.frontier_size,
        s->X, s->target, s->mask, s->nrow, s->ncol, &s->params,
        s->seed_slopes, s->seed_intercepts, s->seed_centers, s->seed_constant) != 0) return -1;
    for (int32_t i = 0; i < m->n_frontier; i++) {
        add_to_complexity_hof(s->hof, &s->n_hof, s->params.complexity_hof_size, &m->frontier[i], &s->params);
    }
    sort_frontier(m->frontier, m->n_frontier);
    if (select_final_formula(&m->formulas[s->cls], m->frontier, m->n_frontier,
        s->hof, s->n_hof, &s->params) != 0) return -1;
    sort_frontier(m->frontier, m->n_frontier);
    if (s->params.task == SYM_TASK_CLASSIFICATION && ++s->cls < m->n_outputs) {
        search_start_tree(s);
        return 0;
    }
    if (s->params.task == SYM_TASK_TRANSFORMER) {
        int32_t k = s->params.top_k < m->n_frontier ? s->params.top_k : m->n_frontier;
        m->n_outputs = m->n_formulas = k;
        for (int32_t i = 0; i < k; i++) {
            sym_tree_formula_free(&m->formulas[i]);
            if (sym_tree_formula_copy(&m->formulas[i], &m->frontier[i]) != 0) return -1;
        }
    }
    s->phase = SEARCH_DONE;
    return 0;
}

int sym_search_propose(sym_search_t *s, const sym_batch_t **out) {
    if (!s || !out) { sym_set_error("search and batch output are required"); return -1; }
    *out = NULL;
    if (s->pending) { *out = &s->batch; return 1; }
    const sym_params_t *p = &s->params;
    sym_model_t *m = s->model;
    for (;;) {
        int32_t count = 0;
        switch (s->phase) {
        case SEARCH_SEED:
            while (s->cursor < p->population && count < s->capacity) {
                int32_t i = s->cursor++;
                int seeded = sym_tree_formula_seed(&s->pop[i], i, s->ncol, &s->active,
                    s->seed_constant, s->seed_slopes, s->seed_intercepts, s->seed_centers);
                if (seeded < 0 || (!seeded && formula_random(&s->pop[i], &s->rng, s->ncol, &s->active) != 0)) goto fail;
                s->views[count++] = &s->pop[i];
            }
            if (count) return search_issue(s, count, 1, SYM_BATCH_SEED, out);
            s->phase = SEARCH_GENERATION;
            break;
        case SEARCH_GENERATION:
            s->cursor = 0;
            if (s->gen >= p->generations) { s->phase = SEARCH_FINAL; break; }
            s->active = *p;
            s->active.max_nodes = current_warmup_max_nodes(p, s->gen);
            if (s->active.max_depth > s->active.max_nodes) s->active.max_depth = s->active.max_nodes;
            s->n_score_rows = build_row_sample(s->score_rows, s->nrow, p->row_sample_size, &s->rng);
            s->phase = s->n_score_rows > 0 ? SEARCH_RESAMPLE : SEARCH_SORT;
            break;
        case SEARCH_RESAMPLE:
        case SEARCH_FINAL:
            while (s->cursor < p->population && count < s->capacity) s->views[count++] = &s->pop[s->cursor++];
            if (count) return search_issue(s, count, s->phase == SEARCH_RESAMPLE,
                s->phase == SEARCH_RESAMPLE ? SYM_BATCH_RESAMPLE : SYM_BATCH_FINAL, out);
            s->phase = s->phase == SEARCH_RESAMPLE ? SEARCH_SORT : SEARCH_FINISH_TREE;
            break;
        case SEARCH_SORT:
            for (int32_t i = 0; i < p->population; i++) s->sorted[i] = &s->pop[i];
            qsort(s->sorted, (size_t)p->population, sizeof(sym_formula_t *), cmp_formula_ptr);
            s->cursor = 0;
            s->phase = SEARCH_FRONTIER;
            break;
        case SEARCH_FRONTIER:
            while (s->cursor < p->frontier_size && s->cursor < p->population && count < s->capacity) {
                if (sym_tree_formula_copy(&s->scratch[count], s->sorted[s->cursor++]) != 0) goto fail;
                s->views[count] = &s->scratch[count];
                count++;
            }
            if (count) return search_issue(s, count, 0, SYM_BATCH_FRONTIER, out);
            s->phase = SEARCH_TOP;
            break;
        case SEARCH_TOP:
            if (sym_tree_formula_copy(&s->scratch[0], s->sorted[0]) != 0) goto fail;
            s->views[0] = &s->scratch[0];
            s->phase = SEARCH_AFTER_TOP;
            return search_issue(s, 1, 0, SYM_BATCH_TOP, out);
        case SEARCH_AFTER_TOP: {
            double objective = s->scratch[0].objective;
            sym_tree_formula_free(&s->scratch[0]);
            if (objective + p->tol < s->best_objective) { s->best_objective = objective; s->stale = 0; }
            else if (++s->stale >= p->early_stop_rounds && p->early_stop_rounds > 0) {
                s->cursor = 0; s->phase = SEARCH_FINAL; break;
            }
            for (int32_t i = 0; i < p->population; i++) sym_tree_formula_free(&s->next[i]);
            sort_frontier(m->frontier, m->n_frontier);
            s->island = 0;
            s->phase = SEARCH_ISLAND;
            break;
        }
        case SEARCH_ISLAND: {
            if (s->island >= p->islands) { s->phase = SEARCH_REFINE; break; }
            s->start = (int32_t)(((int64_t)s->island * p->population) / p->islands);
            s->end = (int32_t)(((int64_t)(s->island + 1) * p->population) / p->islands);
            s->out_i = s->start;
            int32_t size = s->end - s->start;
            int32_t elite = p->islands <= 1 ? p->elite_count : (p->elite_count + p->islands - 1) / p->islands;
            if (elite < 1) elite = 1;
            if (elite > size / 2) elite = size / 2;
            if (elite < 1) elite = 1;
            for (int32_t e = 0; e < elite && s->out_i < s->end; e++) {
                if (copy_best_from_range(&s->next[s->out_i], s->pop, s->start, s->end, e) != 0) break;
                s->out_i++;
            }
            int migration = p->migration_interval > 0 && p->migration_count > 0 && (s->gen + 1) % p->migration_interval == 0;
            int copies = migration && m->n_frontier > 0 ? p->migration_count : p->islands <= 1 ? (p->elite_count < 4 ? p->elite_count : 4) : 0;
            for (int32_t h = 0; h < copies && s->out_i < s->end; h++) {
                int32_t fi = migration && m->n_frontier > 0 ? (s->island * p->migration_count + h) % m->n_frontier : h;
                if (fi >= m->n_frontier) break;
                if (formula_exists_in_slots(s->next + s->start, s->out_i - s->start, &m->frontier[fi])) continue;
                if (sym_tree_formula_copy(&s->next[s->out_i], &m->frontier[fi]) != 0) goto fail;
                s->out_i++;
            }
            s->brood = 0;
            s->phase = SEARCH_CHILDREN;
            break;
        }
        case SEARCH_CHILDREN:
            while (s->out_i < s->end && count < s->capacity) {
                if (make_offspring(&s->scratch[count], s->pop, s->start, s->end, &s->rng, s->ncol, &s->active) != 0) goto fail;
                s->slots[count] = s->out_i;
                s->views[count] = &s->scratch[count];
                count++;
                if (++s->brood == s->active.brood_size) { s->brood = 0; s->out_i++; }
            }
            if (count) return search_issue(s, count, 1, SYM_BATCH_OFFSPRING, out);
            s->island++;
            s->phase = SEARCH_ISLAND;
            break;
        case SEARCH_REFINE:
            if (p->local_refine_count > 0 && p->local_refine_interval > 0 && (s->gen + 1) % p->local_refine_interval == 0) {
                for (int32_t i = 0; i < p->population; i++) s->sorted[i] = &s->next[i];
                qsort(s->sorted, (size_t)p->population, sizeof(sym_formula_t *), cmp_formula_ptr);
                int32_t n = p->local_refine_count < p->population ? p->local_refine_count : p->population;
                for (int32_t i = 0; i < n; i++) {
                    sym_tree_try_affine_rescale(s->sorted[i], s->X, s->target, s->mask, s->nrow, s->ncol, p);
                    sym_tree_improve_constants(s->sorted[i], s->X, s->target, s->mask, s->nrow, s->ncol, p);
                    search_archive(s, s->sorted[i]);
                    eval_formula_loss_search_rows(s->sorted[i], s->X, s->target, s->mask, s->nrow, s->ncol, p,
                        s->score_rows, s->n_score_rows, s->pred_buf, formula_complexity(s->sorted[i]));
                }
            }
            { sym_formula_t *tmp = s->pop; s->pop = s->next; s->next = tmp; }
            s->gen++;
            s->phase = SEARCH_GENERATION;
            break;
        case SEARCH_FINISH_TREE:
            if (search_finish_tree(s) != 0) goto fail;
            break;
        case SEARCH_DONE: return 0;
        case SEARCH_FAILED: sym_set_error("search has failed"); return -1;
        case SEARCH_TAKEN: sym_set_error("search model has already been transferred"); return -1;
        }
    }
fail:
    s->phase = SEARCH_FAILED;
    if (!g_error[0]) sym_set_error("symbolic search failed");
    return -1;
}

const sym_batch_t *sym_search_pending(const sym_search_t *s) {
    return s && s->pending ? &s->batch : NULL;
}

int sym_search_score(sym_search_t *s, uint32_t id, sym_search_score_t *out, int32_t count) {
    if (!s || !s->pending || id != s->batch.id || !out || count != s->batch.count) {
        sym_set_error("score requires the current batch and exact result count"); return -1;
    }
    for (int32_t i = 0; i < count; i++) {
        sym_formula_t f = *s->views[i];
        double complexity = s->base_complexities[i];
        eval_formula_loss_search_rows(&f, s->X, s->target, s->mask, s->nrow, s->ncol, &s->params,
            s->batch.row_indices, s->batch.row_indices ? s->batch.rows : 0, s->pred_buf, complexity);
        out[i].train_loss = f.train_loss; out[i].valid_loss = f.valid_loss;
        out[i].affine = f.complexity != complexity;
    }
    return 0;
}

int sym_search_accept(sym_search_t *s, uint32_t id, const sym_search_score_t *scores, int32_t count) {
    if (!s || !s->pending || id != s->batch.id || !scores || count != s->batch.count) {
        sym_set_error("accept requires the current batch and exact result count"); return -1;
    }
    /* Validate the entire batch before changing any formula or brood winner. */
    for (int32_t i = 0; i < count; i++) {
        const sym_formula_t *f = s->views[i];
        if (!sym_tree_is_finite_value(scores[i].train_loss) || !sym_tree_is_finite_value(scores[i].valid_loss) ||
            scores[i].train_loss < 0 || scores[i].valid_loss < 0 ||
            (scores[i].affine != 0 && scores[i].affine != 1) ||
            (scores[i].affine && (s->params.loss != SYM_LOSS_MSE || f->n_nodes + 4 > f->capacity))) {
            sym_set_error("invalid loss or affine flag in score batch"); return -1;
        }
    }
    for (int32_t i = 0; i < count; i++) {
        sym_formula_t *f = (sym_formula_t *)s->views[i];
        f->train_loss = scores[i].train_loss; f->valid_loss = scores[i].valid_loss;
        f->complexity = s->base_complexities[i];
        if (scores[i].affine) f->complexity += op_cost(SYM_OP_CONST) + op_cost(SYM_OP_MUL) + op_cost(SYM_OP_CONST) + op_cost(SYM_OP_ADD);
        f->objective = f->valid_loss + s->params.complexity_penalty * f->complexity;
        if (!sym_tree_is_finite_value(f->objective)) f->objective = DBL_MAX;
        if (s->batch.stage == SYM_BATCH_FRONTIER) { search_archive(s, f); sym_tree_formula_free(f); }
        else if (s->batch.stage == SYM_BATCH_FINAL) {
            add_to_complexity_hof(s->hof, &s->n_hof, s->params.complexity_hof_size, f, &s->params);
        } else if (s->batch.stage == SYM_BATCH_OFFSPRING) {
            sym_formula_t *best = &s->next[s->slots[i]];
            if (!best->nodes || f->objective < best->objective ||
                (f->objective == best->objective && f->complexity < best->complexity)) {
                sym_tree_formula_free(best); *best = *f; memset(f, 0, sizeof(*f));
            }
            sym_tree_formula_free(f);
        }
    }
    s->pending = 0;
    return 0;
}

sym_model_t *sym_search_finish(sym_search_t *s) {
    if (!s || s->phase != SEARCH_DONE || s->pending) { sym_set_error("search is not finished or model already transferred"); return NULL; }
    sym_model_t *m = s->model;
    s->model = NULL;
    s->phase = SEARCH_TAKEN;
    return m;
}

static void model_free_contents(sym_model_t *m) {
    if (!m) return;
    if (m->formulas) {
        for (int32_t i = 0; i < m->n_formulas; i++) sym_tree_formula_free(&m->formulas[i]);
        free(m->formulas);
    }
    if (m->frontier) {
        for (int32_t i = 0; i < m->n_frontier; i++) sym_tree_formula_free(&m->frontier[i]);
        free(m->frontier);
    }
    m->formulas = NULL;
    m->frontier = NULL;
    m->n_formulas = 0;
    m->n_frontier = 0;
}

void sym_free(sym_model_t *model) {
    if (!model) return;
    model_free_contents(model);
    free(model);
}

sym_model_t *sym_fit(const double *X, int32_t nrow, int32_t ncol, const double *y, const sym_params_t *params) {
    sym_search_t *s = sym_search_new(X, nrow, ncol, y, params, 64);
    if (!s) return NULL;
    const sym_batch_t *batch;
    sym_search_score_t scores[64];
    int rc;
    while ((rc = sym_search_propose(s, &batch)) > 0) {
        if (sym_search_score(s, batch->id, scores, batch->count) != 0 ||
            sym_search_accept(s, batch->id, scores, batch->count) != 0) { rc = -1; break; }
    }
    sym_model_t *model = rc == 0 ? sym_search_finish(s) : NULL;
    sym_search_free(s);
    return model;
}

int sym_predict_raw(const sym_model_t *model, const double *X, int32_t nrow, int32_t ncol, double *out) {
    if (!model || !X || !out || nrow < 0 || ncol != model->n_features) {
        sym_set_error("invalid predict_raw arguments");
        return -1;
    }
    for (int32_t i = 0; i < nrow; i++) {
        const double *row = X + (size_t)i * ncol;
        for (int32_t j = 0; j < model->n_outputs; j++) {
            out[(size_t)i * model->n_outputs + j] = sym_tree_formula_eval_row(&model->formulas[j], row, ncol);
        }
    }
    return 0;
}

int sym_transform(const sym_model_t *model, const double *X, int32_t nrow, int32_t ncol, double *out) {
    return sym_predict_raw(model, X, nrow, ncol, out);
}

int sym_predict_proba(const sym_model_t *model, const double *X, int32_t nrow, int32_t ncol, double *out) {
    if (!model || !X || !out || model->task != SYM_TASK_CLASSIFICATION || ncol != model->n_features) {
        sym_set_error("predict_proba requires a fitted classifier and matching feature count");
        return -1;
    }
    if (model->n_classes <= 2) {
        for (int32_t i = 0; i < nrow; i++) {
            double s = sym_tree_formula_eval_row(&model->formulas[0], X + (size_t)i * ncol, ncol);
            double p = 1.0 / (1.0 + exp(-sym_tree_clamp_double(s, -60.0, 60.0)));
            out[(size_t)i * 2] = 1.0 - p;
            out[(size_t)i * 2 + 1] = p;
        }
    } else {
        double *scores = (double *)malloc((size_t)model->n_classes * sizeof(double));
        if (!scores) {
            sym_set_error("out of memory in predict_proba");
            return -1;
        }
        for (int32_t i = 0; i < nrow; i++) {
            const double *row = X + (size_t)i * ncol;
            double max_s = -DBL_MAX;
            for (int32_t c = 0; c < model->n_classes; c++) {
                scores[c] = sym_tree_formula_eval_row(&model->formulas[c], row, ncol);
                if (scores[c] > max_s) max_s = scores[c];
            }
            double denom = 0.0;
            for (int32_t c = 0; c < model->n_classes; c++) {
                scores[c] = exp(sym_tree_clamp_double(scores[c] - max_s, -60.0, 60.0));
                denom += scores[c];
            }
            if (denom <= 0.0 || !sym_tree_is_finite_value(denom)) denom = 1.0;
            for (int32_t c = 0; c < model->n_classes; c++) {
                out[(size_t)i * model->n_classes + c] = scores[c] / denom;
            }
        }
        free(scores);
    }
    return 0;
}

int sym_predict(const sym_model_t *model, const double *X, int32_t nrow, int32_t ncol, double *out) {
    if (!model || !X || !out || ncol != model->n_features) {
        sym_set_error("invalid predict arguments");
        return -1;
    }
    if (model->task == SYM_TASK_CLASSIFICATION) {
        double *proba = (double *)malloc((size_t)nrow * model->n_classes * sizeof(double));
        if (!proba) {
            sym_set_error("out of memory in predict");
            return -1;
        }
        if (sym_predict_proba(model, X, nrow, ncol, proba) != 0) {
            free(proba);
            return -1;
        }
        for (int32_t i = 0; i < nrow; i++) {
            int32_t best = 0;
            double best_p = proba[(size_t)i * model->n_classes];
            for (int32_t c = 1; c < model->n_classes; c++) {
                double p = proba[(size_t)i * model->n_classes + c];
                if (p > best_p) {
                    best_p = p;
                    best = c;
                }
            }
            out[i] = (double)best;
        }
        free(proba);
        return 0;
    }
    if (model->task == SYM_TASK_TRANSFORMER) {
        return sym_transform(model, X, nrow, ncol, out);
    }
    for (int32_t i = 0; i < nrow; i++) {
        out[i] = sym_tree_formula_eval_row(&model->formulas[0], X + (size_t)i * ncol, ncol);
    }
    return 0;
}

double sym_score(const sym_model_t *model, const double *X, int32_t nrow, int32_t ncol, const double *y) {
    if (!model || !X || !y || nrow <= 0) {
        sym_set_error("invalid score arguments");
        return NAN;
    }
    double *pred = (double *)malloc((size_t)nrow * sizeof(double));
    if (!pred) {
        sym_set_error("out of memory in score");
        return NAN;
    }
    if (sym_predict(model, X, nrow, ncol, pred) != 0) {
        free(pred);
        return NAN;
    }
    double score = 0.0;
    if (model->task == SYM_TASK_CLASSIFICATION) {
        int32_t ok = 0;
        for (int32_t i = 0; i < nrow; i++) {
            if ((int32_t)llround(pred[i]) == (int32_t)llround(y[i])) ok++;
        }
        score = (double)ok / (double)nrow;
    } else {
        double mean = 0.0;
        for (int32_t i = 0; i < nrow; i++) mean += y[i];
        mean /= (double)nrow;
        double ss_res = 0.0;
        double ss_tot = 0.0;
        for (int32_t i = 0; i < nrow; i++) {
            double e = y[i] - pred[i];
            double t = y[i] - mean;
            ss_res += e * e;
            ss_tot += t * t;
        }
        score = ss_tot > 0.0 ? 1.0 - ss_res / ss_tot : 0.0;
    }
    free(pred);
    return score;
}

/*
 * sym.h -- symbolic models for wlearn (C11 core)
 *
 * The core owns the symbolic formula IR, evolutionary search, prediction,
 * transformation, and raw binary serialization. JS/Python wrappers only
 * normalize matrices, package WLRN bundles, and register loaders.
 */

#ifndef WLEARN_SYM_H
#define WLEARN_SYM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SYM_TASK_REGRESSION = 0,
    SYM_TASK_CLASSIFICATION = 1,
    SYM_TASK_TRANSFORMER = 2
} sym_task_t;

typedef enum {
    SYM_LOSS_MSE = 0,
    SYM_LOSS_MAE = 1,
    SYM_LOSS_HUBER = 2,
    SYM_LOSS_LOGLOSS = 3
} sym_loss_t;

typedef enum {
    SYM_OP_CONST = 0,
    SYM_OP_VAR = 1,
    SYM_OP_ADD = 2,
    SYM_OP_SUB = 3,
    SYM_OP_MUL = 4,
    SYM_OP_DIV = 5,
    SYM_OP_NEG = 6,
    SYM_OP_ABS = 7,
    SYM_OP_SQRT = 8,
    SYM_OP_LOG = 9,
    SYM_OP_EXP = 10,
    SYM_OP_SIN = 11,
    SYM_OP_COS = 12,
    SYM_OP_TANH = 13,
    SYM_OP_MIN = 14,
    SYM_OP_MAX = 15
} sym_op_t;

typedef enum {
    SYM_OPSET_BASIC = 0,
    SYM_OPSET_SMOOTH = 1,
    SYM_OPSET_FULL = 2
} sym_opset_t;

typedef enum {
    SYM_FINAL_OBJECTIVE = 0,
    SYM_FINAL_LOSS = 1,
    SYM_FINAL_SCORE = 2
} sym_final_selector_t;

typedef struct {
    int32_t op;
    int32_t left;
    int32_t right;
    int32_t feature;
    double value;
} sym_node_t;

typedef struct {
    sym_node_t *nodes;
    int32_t n_nodes;
    int32_t capacity;
    double train_loss;
    double valid_loss;
    double objective;
    double complexity;
} sym_formula_t;

typedef struct {
    int32_t task;
    int32_t population;
    int32_t generations;
    int32_t max_nodes;
    int32_t max_depth;
    int32_t frontier_size;
    int32_t tournament_size;
    int32_t elite_count;
    int32_t top_k;
    int32_t n_classes;
    int32_t loss;
    int32_t operator_set;
    int32_t early_stop_rounds;
    uint32_t seed;
    double validation_fraction;
    double complexity_penalty;
    double mutation_rate;
    double crossover_rate;
    double constant_rate;
    double const_min;
    double const_max;
    double huber_delta;
    double tol;
    int32_t islands;
    int32_t migration_interval;
    int32_t migration_count;
    int32_t warmup_generations;
    int32_t warmup_min_nodes;
    int32_t brood_size;
    int32_t row_sample_size;
    int32_t local_refine_interval;
    int32_t local_refine_count;
    int32_t complexity_hof_size;
    int32_t final_selector;
    double complexity_bucket_width;
} sym_params_t;

typedef struct {
    int32_t task;
    int32_t n_features;
    int32_t n_classes;
    int32_t n_outputs;
    uint32_t seed;
    sym_params_t params;
    sym_formula_t *formulas;
    int32_t n_formulas;
    sym_formula_t *frontier;
    int32_t n_frontier;
} sym_model_t;

void sym_params_init(sym_params_t *params);

sym_model_t *sym_fit(
    const double *X,
    int32_t nrow,
    int32_t ncol,
    const double *y,
    const sym_params_t *params
);

/* Owned training buffers and search state. Views returned by propose remain
 * valid only until accept/free; callers must copy before asynchronous use. */
typedef struct sym_search sym_search_t;
typedef enum {
    SYM_BATCH_SEED, SYM_BATCH_RESAMPLE, SYM_BATCH_FRONTIER,
    SYM_BATCH_TOP, SYM_BATCH_OFFSPRING, SYM_BATCH_FINAL
} sym_batch_stage_t;
typedef struct {
    uint32_t id;
    int32_t stage, count, nrow, ncol, rows;
    const sym_formula_t *const *formulas;
    const double *X, *target;
    const uint8_t *validation_mask;
    const int32_t *row_indices;
} sym_batch_t;
typedef struct {
    double train_loss, valid_loss;
    int32_t affine;
} sym_search_score_t;
sym_search_t *sym_search_new(const double *X, int32_t nrow, int32_t ncol,
    const double *y, const sym_params_t *params, int32_t batch_capacity);
/* 1: pending batch (repeated propose returns it); 0: complete; -1: error. */
int sym_search_propose(sym_search_t *s, const sym_batch_t **out);
const sym_batch_t *sym_search_pending(const sym_search_t *s);
int sym_search_score(sym_search_t *s, uint32_t id, sym_search_score_t *out, int32_t count);
int sym_search_accept(sym_search_t *s, uint32_t id, const sym_search_score_t *scores, int32_t count);
/* Transfers model ownership exactly once, after propose returns 0. */
sym_model_t *sym_search_finish(sym_search_t *s);
void sym_search_free(sym_search_t *s);

int sym_predict(
    const sym_model_t *model,
    const double *X,
    int32_t nrow,
    int32_t ncol,
    double *out
);

int sym_predict_raw(
    const sym_model_t *model,
    const double *X,
    int32_t nrow,
    int32_t ncol,
    double *out
);

int sym_predict_proba(
    const sym_model_t *model,
    const double *X,
    int32_t nrow,
    int32_t ncol,
    double *out
);

int sym_transform(
    const sym_model_t *model,
    const double *X,
    int32_t nrow,
    int32_t ncol,
    double *out
);

double sym_score(
    const sym_model_t *model,
    const double *X,
    int32_t nrow,
    int32_t ncol,
    const double *y
);

int sym_save(const sym_model_t *model, char **out_buf, int32_t *out_len);
sym_model_t *sym_load(const char *buf, int32_t len);

int sym_formula_text(
    const sym_model_t *model,
    int32_t formula_index,
    char **out_buf,
    int32_t *out_len
);

int sym_formula_json(
    const sym_model_t *model,
    int32_t formula_index,
    char **out_buf,
    int32_t *out_len
);

int sym_set_formula_constant(
    sym_model_t *model,
    int32_t formula_index,
    int32_t node_index,
    double value
);

int sym_set_formula_metrics(
    sym_model_t *model,
    int32_t formula_index,
    double train_loss,
    double valid_loss,
    double objective
);

const char *sym_get_error(void);
void sym_free(sym_model_t *model);
void sym_free_buffer(void *ptr);

#ifdef __cplusplus
}
#endif

#endif

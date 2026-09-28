#include "sym.h"
#include "sym_family.h"
#include "sym_internal.h"
#include <math.h>

#include <stdint.h>

const char *wl_sym_get_last_error(void) {
    return sym_get_error();
}

static sym_params_t
wl_sym_params(int task, int n_classes, int top_k, int population, int generations, int max_nodes,
              int max_depth, int frontier_size, int tournament_size, int elite_count, int loss,
              int operator_set, int early_stop_rounds, int seed, double validation_fraction,
              double complexity_penalty, double mutation_rate, double crossover_rate,
              double constant_rate, double const_min, double const_max, double huber_delta,
              double tol, int islands, int migration_interval, int migration_count,
              int warmup_generations, int warmup_min_nodes, int brood_size, int row_sample_size,
              int local_refine_interval, int local_refine_count, int complexity_hof_size,
              int final_selector, double complexity_bucket_width) {
    sym_params_t params;
    sym_params_init(&params);
    params.task = task;
    params.n_classes = n_classes;
    params.top_k = top_k;
    params.population = population;
    params.generations = generations;
    params.max_nodes = max_nodes;
    params.max_depth = max_depth;
    params.frontier_size = frontier_size;
    params.tournament_size = tournament_size;
    params.elite_count = elite_count;
    params.loss = loss;
    params.operator_set = operator_set;
    params.early_stop_rounds = early_stop_rounds;
    params.seed = (uint32_t)seed;
    params.validation_fraction = validation_fraction;
    params.complexity_penalty = complexity_penalty;
    params.mutation_rate = mutation_rate;
    params.crossover_rate = crossover_rate;
    params.constant_rate = constant_rate;
    params.const_min = const_min;
    params.const_max = const_max;
    params.huber_delta = huber_delta;
    params.tol = tol;
    params.islands = islands;
    params.migration_interval = migration_interval;
    params.migration_count = migration_count;
    params.warmup_generations = warmup_generations;
    params.warmup_min_nodes = warmup_min_nodes;
    params.brood_size = brood_size;
    params.row_sample_size = row_sample_size;
    params.local_refine_interval = local_refine_interval;
    params.local_refine_count = local_refine_count;
    params.complexity_hof_size = complexity_hof_size;
    params.final_selector = final_selector;
    params.complexity_bucket_width = complexity_bucket_width;
    return params;
}

sym_model_t *wl_sym_fit(const double *X, int nrow, int ncol, const double *y, int task,
                        int n_classes, int top_k, int population, int generations, int max_nodes,
                        int max_depth, int frontier_size, int tournament_size, int elite_count,
                        int loss, int operator_set, int early_stop_rounds, int seed,
                        double validation_fraction, double complexity_penalty, double mutation_rate,
                        double crossover_rate, double constant_rate, double const_min,
                        double const_max, double huber_delta, double tol, int islands,
                        int migration_interval, int migration_count, int warmup_generations,
                        int warmup_min_nodes, int brood_size, int row_sample_size,
                        int local_refine_interval, int local_refine_count, int complexity_hof_size,
                        int final_selector, double complexity_bucket_width) {
    sym_params_t params = wl_sym_params(
        task, n_classes, top_k, population, generations, max_nodes, max_depth, frontier_size,
        tournament_size, elite_count, loss, operator_set, early_stop_rounds, seed,
        validation_fraction, complexity_penalty, mutation_rate, crossover_rate, constant_rate,
        const_min, const_max, huber_delta, tol, islands, migration_interval, migration_count,
        warmup_generations, warmup_min_nodes, brood_size, row_sample_size, local_refine_interval,
        local_refine_count, complexity_hof_size, final_selector, complexity_bucket_width);
    return sym_fit(X, nrow, ncol, y, &params);
}

sym_search_t *
wl_sym_search_new(const double *X, int nrow, int ncol, const double *y, int task, int n_classes,
                  int top_k, int population, int generations, int max_nodes, int max_depth,
                  int frontier_size, int tournament_size, int elite_count, int loss,
                  int operator_set, int early_stop_rounds, int seed, double validation_fraction,
                  double complexity_penalty, double mutation_rate, double crossover_rate,
                  double constant_rate, double const_min, double const_max, double huber_delta,
                  double tol, int islands, int migration_interval, int migration_count,
                  int warmup_generations, int warmup_min_nodes, int brood_size, int row_sample_size,
                  int local_refine_interval, int local_refine_count, int complexity_hof_size,
                  int final_selector, double complexity_bucket_width) {
    sym_params_t params = wl_sym_params(
        task, n_classes, top_k, population, generations, max_nodes, max_depth, frontier_size,
        tournament_size, elite_count, loss, operator_set, early_stop_rounds, seed,
        validation_fraction, complexity_penalty, mutation_rate, crossover_rate, constant_rate,
        const_min, const_max, huber_delta, tol, islands, migration_interval, migration_count,
        warmup_generations, warmup_min_nodes, brood_size, row_sample_size, local_refine_interval,
        local_refine_count, complexity_hof_size, final_selector, complexity_bucket_width);
    return sym_search_new(X, nrow, ncol, y, &params, 64);
}

int wl_sym_predict(const sym_model_t *model, const double *X, int nrow, int ncol, double *out) {
    return sym_predict(model, X, (int32_t)nrow, (int32_t)ncol, out);
}

int wl_sym_predict_raw(const sym_model_t *model, const double *X, int nrow, int ncol, double *out) {
    return sym_predict_raw(model, X, (int32_t)nrow, (int32_t)ncol, out);
}

int wl_sym_predict_proba(const sym_model_t *model, const double *X, int nrow, int ncol,
                         double *out) {
    return sym_predict_proba(model, X, (int32_t)nrow, (int32_t)ncol, out);
}

int wl_sym_transform(const sym_model_t *model, const double *X, int nrow, int ncol, double *out) {
    return sym_transform(model, X, (int32_t)nrow, (int32_t)ncol, out);
}

double wl_sym_score(const sym_model_t *model, const double *X, int nrow, int ncol,
                    const double *y) {
    return sym_score(model, X, (int32_t)nrow, (int32_t)ncol, y);
}

int wl_sym_save(const sym_model_t *model, char **out_buf, int *out_len) {
    int32_t len32 = 0;
    int rc = sym_save(model, out_buf, &len32);
    if (rc == 0 && out_len)
        *out_len = (int)len32;
    return rc;
}

sym_model_t *wl_sym_load(const char *buf, int len) {
    return sym_load(buf, (int32_t)len);
}

void wl_sym_free(sym_model_t *model) {
    sym_free(model);
}

void wl_sym_free_buffer(void *ptr) {
    sym_free_buffer(ptr);
}

int wl_sym_formula_text(const sym_model_t *model, int formula_index, char **out_buf, int *out_len) {
    int32_t len32 = 0;
    int rc = sym_formula_text(model, (int32_t)formula_index, out_buf, &len32);
    if (rc == 0 && out_len)
        *out_len = (int)len32;
    return rc;
}

int wl_sym_formula_json(const sym_model_t *model, int formula_index, char **out_buf, int *out_len) {
    int32_t len32 = 0;
    int rc = sym_formula_json(model, (int32_t)formula_index, out_buf, &len32);
    if (rc == 0 && out_len)
        *out_len = (int)len32;
    return rc;
}

int wl_sym_set_formula_constant(sym_model_t *model, int formula_index, int node_index,
                                double value) {
    return sym_set_formula_constant(model, (int32_t)formula_index, (int32_t)node_index, value);
}

int wl_sym_set_formula_metrics(sym_model_t *model, int formula_index, double train_loss,
                               double valid_loss, double objective) {
    return sym_set_formula_metrics(model, (int32_t)formula_index, train_loss, valid_loss,
                                   objective);
}

int wl_sym_get_task(const sym_model_t *model) {
    return model ? model->task : -1;
}

int wl_sym_get_n_features(const sym_model_t *model) {
    return model ? model->n_features : 0;
}

int wl_sym_get_n_classes(const sym_model_t *model) {
    return model ? model->n_classes : 0;
}

int wl_sym_get_n_outputs(const sym_model_t *model) {
    return model ? model->n_outputs : 0;
}

int wl_sym_get_n_formulas(const sym_model_t *model) {
    return model ? model->n_formulas : 0;
}

int wl_sym_get_frontier_size(const sym_model_t *model) {
    return model ? model->n_frontier : 0;
}

int wl_sym_get_formula_n_nodes(const sym_model_t *model, int formula_index) {
    if (!model || formula_index < 0)
        return 0;
    if (formula_index < model->n_formulas)
        return model->formulas[formula_index].n_nodes;
    formula_index -= model->n_formulas;
    if (formula_index < model->n_frontier)
        return model->frontier[formula_index].n_nodes;
    return 0;
}

double wl_sym_get_formula_train_loss(const sym_model_t *model, int formula_index) {
    if (!model || formula_index < 0)
        return 0.0;
    if (formula_index < model->n_formulas)
        return model->formulas[formula_index].train_loss;
    formula_index -= model->n_formulas;
    if (formula_index < model->n_frontier)
        return model->frontier[formula_index].train_loss;
    return 0.0;
}

double wl_sym_get_formula_valid_loss(const sym_model_t *model, int formula_index) {
    if (!model || formula_index < 0)
        return 0.0;
    if (formula_index < model->n_formulas)
        return model->formulas[formula_index].valid_loss;
    formula_index -= model->n_formulas;
    if (formula_index < model->n_frontier)
        return model->frontier[formula_index].valid_loss;
    return 0.0;
}

double wl_sym_get_formula_objective(const sym_model_t *model, int formula_index) {
    if (!model || formula_index < 0)
        return 0.0;
    if (formula_index < model->n_formulas)
        return model->formulas[formula_index].objective;
    formula_index -= model->n_formulas;
    if (formula_index < model->n_frontier)
        return model->frontier[formula_index].objective;
    return 0.0;
}

double wl_sym_get_formula_complexity(const sym_model_t *model, int formula_index) {
    if (!model || formula_index < 0)
        return 0.0;
    if (formula_index < model->n_formulas)
        return model->formulas[formula_index].complexity;
    formula_index -= model->n_formulas;
    if (formula_index < model->n_frontier)
        return model->frontier[formula_index].complexity;
    return 0.0;
}

/* Frontends exchange scores as packed doubles, independent of C struct padding. */
int wl_sym_search_propose(sym_search_t *s) {
    const sym_batch_t *b;
    int rc = sym_search_propose(s, &b);
    return rc > 0 ? b->count : rc;
}
uint32_t wl_sym_search_batch_id(sym_search_t *s) {
    const sym_batch_t *b = sym_search_pending(s);
    return b ? b->id : 0;
}
int wl_sym_search_score(sym_search_t *s, uint32_t id, double *out, int count) {
    sym_search_score_t scores[64];
    if (!out || count < 1 || count > 64)
        return sym_search_score(s, id, NULL, count);
    int rc = sym_search_score(s, id, scores, count);
    if (rc)
        return rc;
    for (int i = 0; i < count; i++) {
        out[3 * i] = scores[i].train_loss;
        out[3 * i + 1] = scores[i].valid_loss;
        out[3 * i + 2] = scores[i].affine;
    }
    return 0;
}
int wl_sym_search_accept(sym_search_t *s, uint32_t id, const double *values, int count) {
    sym_search_score_t scores[64];
    if (!values || count < 1 || count > 64)
        return sym_search_accept(s, id, NULL, count);
    for (int i = 0; i < count; i++) {
        scores[i].train_loss = values[3 * i];
        scores[i].valid_loss = values[3 * i + 1];
        /* Avoid undefined float-to-integer conversion on malformed input. */
        scores[i].affine = values[3 * i + 2] == 0 ? 0 : values[3 * i + 2] == 1 ? 1 : -1;
    }
    return sym_search_accept(s, id, scores, count);
}
sym_model_t *wl_sym_search_finish(sym_search_t *s) {
    return sym_search_finish(s);
}
void wl_sym_search_free(sym_search_t *s) {
    sym_search_free(s);
}

/* These are borrowed views of the pending batch. Frontends must copy views
 * before yielding or calling an operation that may grow Wasm memory. */
int wl_sym_search_batch_stage(const sym_search_t *s) {
    const sym_batch_t *b = sym_search_pending(s);
    return b ? b->stage : -1;
}
int wl_sym_search_batch_rows(const sym_search_t *s) {
    const sym_batch_t *b = sym_search_pending(s);
    return b ? b->rows : 0;
}
const int32_t *wl_sym_search_batch_row_indices(const sym_search_t *s) {
    const sym_batch_t *b = sym_search_pending(s);
    return b ? b->row_indices : NULL;
}
const double *wl_sym_search_batch_X(const sym_search_t *s) {
    const sym_batch_t *b = sym_search_pending(s);
    return b ? b->X : NULL;
}
const double *wl_sym_search_batch_target(const sym_search_t *s) {
    const sym_batch_t *b = sym_search_pending(s);
    return b ? b->target : NULL;
}
const uint8_t *wl_sym_search_batch_mask(const sym_search_t *s) {
    const sym_batch_t *b = sym_search_pending(s);
    return b ? b->validation_mask : NULL;
}
int wl_sym_search_batch_node_count(const sym_search_t *s, int index) {
    const sym_batch_t *b = sym_search_pending(s);
    return b && index >= 0 && index < b->count ? b->formulas[index]->n_nodes : -1;
}
/* Four packed int32 fields and one double per node avoid frontend assumptions
 * about sym_node_t padding. Return the exact required node count when out=NULL. */
int wl_sym_search_batch_nodes(const sym_search_t *s, int index, int32_t *out, double *values,
                              int capacity) {
    const sym_batch_t *b = sym_search_pending(s);
    if (!b || index < 0 || index >= b->count)
        return -1;
    const sym_formula_t *f = b->formulas[index];
    if (!out && !values)
        return f->n_nodes;
    if (!out || !values || capacity < f->n_nodes)
        return -1;
    for (int32_t i = 0; i < f->n_nodes; i++) {
        out[4 * i] = f->nodes[i].op;
        out[4 * i + 1] = f->nodes[i].left;
        out[4 * i + 2] = f->nodes[i].right;
        out[4 * i + 3] = f->nodes[i].feature;
        values[i] = f->nodes[i].value;
    }
    return f->n_nodes;
}

/* Packed config: ten integer fields, then seven floating fields. Frontends can
 * pass one contiguous buffer without depending on native structure padding. */
static int wl_family_config(const double *config, int count, sym_family_params_t *out) {
    if (!config || (count < 17 || count > 19)) {
        sym_set_error("invalid family config length");
        return -1;
    }
    for (int i = 0; i < 17; i++) {
        if (!isfinite(config[i]) || (i < 10 && (floor(config[i]) != config[i] || config[i] < 0 ||
                                                config[i] > (i == 7 ? UINT32_MAX : INT32_MAX)))) {
            sym_set_error("invalid family config value");
            return -1;
        }
    }
    sym_family_params_t p;
    sym_family_params_init(&p);
    p.population = (int32_t)config[0];
    p.generations = (int32_t)config[1];
    p.terms = (int32_t)config[2];
    p.elite = (int32_t)config[3];
    p.islands = (int32_t)config[4];
    p.frontier = (int32_t)config[5];
    p.operators = (uint32_t)config[6];
    p.seed = (uint32_t)config[7];
    p.polish = (int32_t)config[8];
    p.patience = (int32_t)config[9];
    p.validation_fraction = config[10];
    p.complexity_penalty = config[11];
    p.immigrant_rate = config[12];
    p.mutation_rate = config[13];
    p.crossover_rate = config[14];
    p.ridge = config[15];
    p.tol = config[16];
    if (count >= 18) {
        if (!isfinite(config[17]) || floor(config[17]) != config[17] || config[17] < 0 ||
            config[17] > 512) {
            sym_set_error("invalid family polish batch size");
            return -1;
        }
        p.polish_batch = (int32_t)config[17];
    }
    if (count == 19) {
        if (config[18] != 0 && config[18] != 1) {
            sym_set_error("hierarchical must be boolean");
            return -1;
        }
        p.hierarchical = (int32_t)config[18];
    }
    *out = p;
    return 0;
}

sym_family_model_t *wl_sym_family_fit(const double *X, int rows, int cols, const double *y,
                                      int task, int classes, const double *config, int count) {
    sym_family_params_t p;
    if (wl_family_config(config, count, &p))
        return NULL;
    return sym_family_fit(X, rows, cols, y, task, classes, &p);
}
sym_family_search_t *wl_sym_family_search_new(const double *X, int rows, int cols, const double *y,
                                              int task, int classes, const double *config,
                                              int count, int capacity, int tile_rows) {
    sym_family_params_t p;
    if (wl_family_config(config, count, &p))
        return NULL;
    return sym_family_search_new(X, rows, cols, y, task, classes, &p, capacity, tile_rows);
}
int wl_sym_family_search_propose(sym_family_search_t *s) {
    const sym_family_batch_t *b;
    return sym_family_search_propose(s, &b);
}
uint32_t wl_sym_family_batch_id(const sym_family_search_t *s) {
    const sym_family_batch_t *b = sym_family_search_pending(s);
    return b ? b->id : 0;
}
int wl_sym_family_batch_shape(const sym_family_search_t *s, int32_t *out, int count) {
    const sym_family_batch_t *b = sym_family_search_pending(s);
    if (!b || !out || count != 8) {
        sym_set_error("invalid family batch shape request");
        return -1;
    }
    out[0] = b->stage;
    out[1] = b->candidates;
    out[2] = b->rows;
    out[3] = b->cols;
    out[4] = b->terms;
    out[5] = b->row_start;
    out[6] = b->head;
    out[7] = b->generation;
    return 0;
}
int wl_sym_family_batch_descriptors(const sym_family_search_t *s, double *out, int count) {
    const sym_family_batch_t *b = sym_family_search_pending(s);
    if (!b || !out || count != b->candidates * b->terms * 5) {
        sym_set_error("invalid family descriptor request");
        return -1;
    }
    for (int i = 0; i < count; i++)
        out[i] = b->descriptors[i];
    return 0;
}

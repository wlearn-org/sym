#include "sym.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        return 0; \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    tests_run++; \
    printf("  %s...", #fn); \
    if (fn()) { tests_passed++; printf(" OK\n"); } else { printf("\n"); } \
} while (0)

static void make_regression(double *X, double *y, int n) {
    for (int i = 0; i < n; i++) {
        double x0 = -2.0 + 4.0 * (double)i / (double)(n - 1);
        double x1 = sin((double)i * 0.37);
        X[i * 2] = x0;
        X[i * 2 + 1] = x1;
        y[i] = 2.0 * x0 - 0.5 * x1 + 1.0;
    }
}

static void make_classification(double *X, double *y, int n) {
    for (int i = 0; i < n; i++) {
        double x0 = -2.0 + 4.0 * (double)i / (double)(n - 1);
        double x1 = cos((double)i * 0.19);
        X[i * 2] = x0;
        X[i * 2 + 1] = x1;
        y[i] = (x0 + 0.25 * x1 > 0.0) ? 1.0 : 0.0;
    }
}

static sym_params_t tiny_params(int task) {
    sym_params_t p;
    sym_params_init(&p);
    p.task = task;
    p.n_classes = task == SYM_TASK_CLASSIFICATION ? 2 : 0;
    p.population = 80;
    p.generations = 45;
    p.max_nodes = 15;
    p.max_depth = 4;
    p.frontier_size = 8;
    p.elite_count = 4;
    p.tournament_size = 4;
    p.validation_fraction = 0.2;
    p.seed = 123;
    p.complexity_penalty = 0.0001;
    p.operator_set = SYM_OPSET_BASIC;
    return p;
}

static int test_regression_fit_predict_save_load(void) {
    enum { N = 80 };
    double X[N * 2];
    double y[N];
    make_regression(X, y, N);
    sym_params_t p = tiny_params(SYM_TASK_REGRESSION);

    sym_model_t *m = sym_fit(X, N, 2, y, &p);
    ASSERT(m != NULL, sym_get_error());
    ASSERT(m->n_outputs == 1, "regressor has one output");
    ASSERT(m->n_frontier > 0, "frontier populated");

    double score = sym_score(m, X, N, 2, y);
    ASSERT(score > 0.85, "regression R2 should be high");

    char *text = NULL;
    int32_t text_len = 0;
    ASSERT(sym_formula_text(m, 0, &text, &text_len) == 0, "formula text");
    ASSERT(text_len > 0 && strstr(text, "x") != NULL, "formula mentions feature");
    sym_free_buffer(text);

    char *raw = NULL;
    int32_t raw_len = 0;
    ASSERT(sym_save(m, &raw, &raw_len) == 0, "save");
    ASSERT(raw_len > 32, "raw len");
    sym_model_t *loaded = sym_load(raw, raw_len);
    ASSERT(loaded != NULL, sym_get_error());

    double pred1[N];
    double pred2[N];
    ASSERT(sym_predict(m, X, N, 2, pred1) == 0, "predict original");
    ASSERT(sym_predict(loaded, X, N, 2, pred2) == 0, "predict loaded");
    for (int i = 0; i < N; i++) ASSERT(fabs(pred1[i] - pred2[i]) < 1e-12, "loaded prediction parity");

    sym_free_buffer(raw);
    sym_free(loaded);
    sym_free(m);
    return 1;
}

static int test_formula_constant_update(void) {
    enum { N = 80 };
    double X[N * 2];
    double y[N];
    make_regression(X, y, N);
    sym_params_t p = tiny_params(SYM_TASK_REGRESSION);

    sym_model_t *m = sym_fit(X, N, 2, y, &p);
    ASSERT(m != NULL, sym_get_error());

    int32_t const_idx = -1;
    for (int32_t i = 0; i < m->formulas[0].n_nodes; i++) {
        if (m->formulas[0].nodes[i].op == SYM_OP_CONST) {
            const_idx = i;
            break;
        }
    }
    ASSERT(const_idx >= 0, "formula has a constant node");

    double old = m->formulas[0].nodes[const_idx].value;
    ASSERT(sym_set_formula_constant(m, 0, const_idx, old + 0.125) == 0, "set constant");
    ASSERT(fabs(m->formulas[0].nodes[const_idx].value - (old + 0.125)) < 1e-12, "constant changed");
    ASSERT(sym_set_formula_metrics(m, 0, 1.5, 1.5, 1.6) == 0, "set metrics");
    ASSERT(fabs(m->formulas[0].train_loss - 1.5) < 1e-12, "train loss changed");

    sym_free(m);
    return 1;
}

static int test_classifier_proba(void) {
    enum { N = 90 };
    double X[N * 2];
    double y[N];
    make_classification(X, y, N);
    sym_params_t p = tiny_params(SYM_TASK_CLASSIFICATION);
    p.loss = SYM_LOSS_LOGLOSS;
    p.seed = 321;

    sym_model_t *m = sym_fit(X, N, 2, y, &p);
    ASSERT(m != NULL, sym_get_error());
    ASSERT(m->n_classes == 2, "binary classes");

    double score = sym_score(m, X, N, 2, y);
    ASSERT(score > 0.85, "classification accuracy");

    double proba[N * 2];
    ASSERT(sym_predict_proba(m, X, N, 2, proba) == 0, "predict proba");
    for (int i = 0; i < N; i++) {
        double s = proba[i * 2] + proba[i * 2 + 1];
        ASSERT(fabs(s - 1.0) < 1e-10, "probabilities sum to one");
        ASSERT(proba[i * 2] >= 0.0 && proba[i * 2] <= 1.0, "p0 range");
        ASSERT(proba[i * 2 + 1] >= 0.0 && proba[i * 2 + 1] <= 1.0, "p1 range");
    }

    sym_free(m);
    return 1;
}

static int test_transformer_top_k(void) {
    enum { N = 64 };
    double X[N * 2];
    double y[N];
    make_regression(X, y, N);
    sym_params_t p = tiny_params(SYM_TASK_TRANSFORMER);
    p.top_k = 4;
    p.frontier_size = 8;
    p.seed = 555;

    sym_model_t *m = sym_fit(X, N, 2, y, &p);
    ASSERT(m != NULL, sym_get_error());
    ASSERT(m->n_outputs == 4, "transformer output count");

    double Z[N * 4];
    ASSERT(sym_transform(m, X, N, 2, Z) == 0, "transform");
    for (int i = 0; i < N * 4; i++) ASSERT(isfinite(Z[i]), "finite transformed value");

    sym_free(m);
    return 1;
}

static int test_modular_search_options(void) {
    enum { N = 96 };
    double X[N * 2];
    double y[N];
    make_regression(X, y, N);
    sym_params_t p = tiny_params(SYM_TASK_REGRESSION);
    p.population = 96;
    p.generations = 30;
    p.max_nodes = 21;
    p.max_depth = 5;
    p.frontier_size = 10;
    p.validation_fraction = 0.0;
    p.islands = 3;
    p.migration_interval = 5;
    p.migration_count = 1;
    p.warmup_generations = 12;
    p.warmup_min_nodes = 7;
    p.brood_size = 3;
    p.row_sample_size = 48;
    p.local_refine_interval = 6;
    p.local_refine_count = 3;
    p.complexity_hof_size = 24;
    p.final_selector = SYM_FINAL_LOSS;
    p.complexity_bucket_width = 2.0;

    sym_model_t *m = sym_fit(X, N, 2, y, &p);
    ASSERT(m != NULL, sym_get_error());
    ASSERT(m->params.islands == 3, "islands persisted");
    ASSERT(m->params.brood_size == 3, "brood size persisted");
    ASSERT(m->params.row_sample_size == 48, "row sample size persisted");
    ASSERT(m->params.complexity_hof_size == 24, "complexity HOF persisted");
    ASSERT(m->params.final_selector == SYM_FINAL_LOSS, "final selector persisted");
    ASSERT(m->n_frontier > 0, "frontier populated with modular options");
    ASSERT(sym_score(m, X, N, 2, y) > 0.75, "modular search score");

    char *raw = NULL;
    int32_t raw_len = 0;
    ASSERT(sym_save(m, &raw, &raw_len) == 0, "save modular search");
    sym_model_t *loaded = sym_load(raw, raw_len);
    ASSERT(loaded != NULL, sym_get_error());
    ASSERT(loaded->params.islands == 3, "loaded islands");
    ASSERT(loaded->params.local_refine_count == 3, "loaded local refine count");
    ASSERT(loaded->params.complexity_hof_size == 24, "loaded complexity HOF");
    ASSERT(loaded->params.final_selector == SYM_FINAL_LOSS, "loaded selector");
    sym_free_buffer(raw);
    sym_free(loaded);
    sym_free(m);
    return 1;
}

int main(void) {
    printf("sym C tests\n");
    RUN_TEST(test_regression_fit_predict_save_load);
    RUN_TEST(test_formula_constant_update);
    RUN_TEST(test_classifier_proba);
    RUN_TEST(test_transformer_top_k);
    RUN_TEST(test_modular_search_options);
    printf("%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}

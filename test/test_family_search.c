#include "sym_family.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c)                                                                               \
    do {                                                                                       \
        if (!(c)) {                                                                            \
            fprintf(stderr, "line %d: %s (%s)\n", __LINE__, #c, sym_get_error());              \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)
static double reference_term(const double *d, const double *row) {
    double a = row[(int)d[0]], b = row[(int)d[1]], z = a * d[3] + d[4];
    switch ((int)d[2]) {
    case 0:
        return a + b;
    case 1:
        return a - b;
    case 2:
        return a * b;
    case 3:
        return a / (fabs(b) < 1e-6 ? (b >= 0 ? 1e-6 : -1e-6) : b);
    case 4:
        return sin(z);
    case 5:
        return cos(z);
    case 6:
        return tanh(z);
    case 7:
        return log(fabs(z) + 1e-6);
    case 8:
        return sqrt(fabs(z));
    default:
        return exp(fmax(-6, fmin(6, z)));
    }
}
static char *drive(const double *X, const double *y, int train_rows, int classes,
                   sym_family_params_t *p, int capacity, int rows, int32_t *size) {
    sym_family_search_t *s =
        sym_family_search_new(X, train_rows, 3, y, classes > 0, classes, p, capacity, rows);
    CHECK(s);
    CHECK(sym_family_search_finish(s) == NULL);
    double *features = malloc((size_t)capacity * rows * p->terms * sizeof(double));
    CHECK(features);
    const sym_family_batch_t *batch, *same;
    int rc;
    while ((rc = sym_family_search_propose(s, &batch)) > 0) {
        CHECK(sym_family_search_propose(s, &same) == 1 && same == batch);
        CHECK(batch->candidates <= capacity && batch->rows <= rows);
        int count = batch->candidates * batch->rows * batch->terms;
        CHECK(sym_family_search_score(s, batch->id, features, count) == 0);
        for (int c = 0; c < batch->candidates; c++)
            for (int r = 0; r < batch->rows; r++)
                for (int t = 0; t < batch->terms; t++) {
                    double expected =
                        reference_term(batch->descriptors + ((size_t)c * batch->terms + t) * 5,
                                       batch->X + (size_t)r * batch->cols);
                    size_t at = ((size_t)c * batch->rows + r) * batch->terms + t;
                    CHECK(features[at] == expected);
                    features[at] = expected; /* External evaluator supplies these features. */
                }
        double first = features[0];
        features[0] = NAN;
        CHECK(sym_family_search_accept(s, batch->id, features, count) == -1);
        features[0] = first;
        CHECK(sym_family_search_accept(s, batch->id + 1, features, count) == -1);
        uint32_t id = batch->id;
        CHECK(sym_family_search_accept(s, id, features, count) == 0);
        CHECK(sym_family_search_accept(s, id, features, count) == -1);
    }
    CHECK(rc == 0);
    if (!p->polish && p->islands == 1 && !p->patience) {
        uint64_t per_head = p->population + (p->generations - 1) * (p->population - p->elite);
        CHECK(sym_family_search_evaluations(s) == per_head * (classes > 2 ? classes : 1));
    }
    sym_family_model_t *m = sym_family_search_finish(s);
    CHECK(m);
    CHECK(sym_family_search_finish(s) == NULL);
    char *bytes = NULL;
    CHECK(sym_family_save(m, &bytes, size) == 0);
    sym_family_free(m);
    sym_family_search_free(s);
    free(features);
    return bytes;
}
/* Exercise full/partial row tiles, candidate tails and the 8 MiB cache bound.
 * The external evaluator never caches: saved bytes must match the C fit path. */
static void cache_boundaries(void) {
    const int counts[] = {127, 128, 129, 4095, 4096, 4097};
    double *X = malloc((size_t)4097 * 3 * sizeof(double));
    double *y = malloc((size_t)4097 * sizeof(double));
    CHECK(X && y);
    for (int r = 0; r < 4097; r++) {
        X[3 * r] = (r % 101 - 50) / 32.;
        X[3 * r + 1] = (r % 23 - 11) / 16.;
        X[3 * r + 2] = (r % 17 - 8) / 8.;
        y[r] = X[3 * r] + .3 * X[3 * r + 1] * X[3 * r + 2];
    }
    sym_family_params_t p;
    sym_family_params_init(&p);
    p.population = 11;
    p.generations = 2;
    p.elite = 2;
    p.terms = 32;
    p.operators = 15;
    p.validation_fraction = .2;
    for (int i = 0; i < 6; i++) {
        sym_family_model_t *m = sym_family_fit(X, counts[i], 3, y, 0, 0, &p);
        CHECK(m);
        char *actual;
        int32_t n, expected_n;
        CHECK(sym_family_save(m, &actual, &n) == 0);
        char *expected = drive(X, y, counts[i], 0, &p, 8, 128, &expected_n);
        CHECK(n == expected_n && memcmp(actual, expected, n) == 0);
        sym_free_buffer(actual);
        sym_free_buffer(expected);
        sym_family_free(m);
    }
    free(X);
    free(y);
}
int main(void) {
    double X[180], y[60];
    for (int i = 0; i < 60; i++) {
        X[3 * i] = (i - 30) / 10.;
        X[3 * i + 1] = sin(i * .47);
        X[3 * i + 2] = cos(i * .31);
    }
    sym_family_params_t p;
    sym_family_params_init(&p);
    p.population = 16;
    p.generations = 3;
    p.terms = 3;
    p.elite = 4;
    p.frontier = 6;
    p.validation_fraction = .2;
    for (int classes = 0; classes <= 3; classes++) {
        if (classes == 1)
            continue;
        for (int i = 0; i < 60; i++)
            y[i] = classes ? i % classes : 2 * X[3 * i] + .3 * X[3 * i + 1] + 1;
        for (int polish = 0; polish <= 1; polish++) {
            p.polish = polish;
            sym_family_model_t *m = sym_family_fit(X, 60, 3, y, classes > 0, classes, &p);
            CHECK(m);
            char *baseline;
            int32_t length;
            CHECK(sym_family_save(m, &baseline, &length) == 0);
            for (int mode = 0; mode < 3; mode++) {
                int32_t n;
                char *bytes =
                    drive(X, y, 60, classes, &p, mode == 0 ? 1 : 7, mode == 1 ? 17 : 60, &n);
                CHECK(n == length && memcmp(bytes, baseline, n) == 0);
                sym_free_buffer(bytes);
            }
            sym_family_model_t *loaded = sym_family_load(baseline, length);
            CHECK(loaded);
            char *again;
            int32_t n;
            CHECK(sym_family_save(loaded, &again, &n) == 0);
            CHECK(n == length && memcmp(again, baseline, n) == 0);
            CHECK(sym_family_load(baseline, length - 1) == NULL);
            for (int at = 4; at <= 44; at += 4) {
                char old[4];
                memcpy(old, baseline + at, 4);
                memset(baseline + at, 255, 4);
                CHECK(sym_family_load(baseline, length) == NULL);
                memcpy(baseline + at, old, 4);
            }
            sym_free_buffer(again);
            sym_free_buffer(baseline);
            sym_family_free(loaded);
            sym_family_free(m);
        }
    }
    for (int i = 0; i < 20; i++) {
        sym_family_search_t *s = sym_family_search_new(X, 60, 3, y, 0, 0, &p, 1, 1);
        CHECK(s);
        const sym_family_batch_t *b;
        CHECK(sym_family_search_propose(s, &b) == 1);
        sym_family_search_free(s);
    }
    cache_boundaries();
    puts(
        "family step: 18 external-evaluator runs, QR tile parity, atomic accept, polish, elite "
        "cache, cancellation, SYM2 and 6 cache-boundary cases passed");
    return 0;
}

#include "sym_family.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#define CHECK(c)                                                                                   \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            fprintf(stderr, "line %d: %s (%s)\n", __LINE__, #c, sym_get_error());                  \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
extern sym_family_model_t *wl_sym_family_fit(const double *, int, int, const double *, int, int,
                                             const double *, int);
int main(void) {
    double X[120], y[60], out[180];
    for (int i = 0; i < 60; i++) {
        X[2 * i] = (i - 30) / 10.0;
        X[2 * i + 1] = sin(i * .47);
        y[i] = 2 * X[2 * i] + .3 * X[2 * i + 1] + 1;
    }
    /* Extended frontend config requests batched nonlinear-parameter polish. */
    double config[] = {24, 2, 3, 4, 1, 4, 1023, 42, 2, 0, .2, 1e-5, 0, .35, .55, 1e-8, 1e-12, 32};
    sym_family_model_t *batched = wl_sym_family_fit(X, 60, 2, y, 0, 0, config, 18);
    CHECK(batched);
    sym_family_free(batched);
    sym_family_params_t p;
    sym_family_params_init(&p);
    CHECK(p.immigrant_rate == 0);
    p.population = 24;
    p.generations = 4;
    p.terms = 3;
    p.frontier = 4;
    p.islands = 3;
    p.operators = 15;
    p.validation_fraction = .2;
    p.polish = 2;
    sym_family_model_t *m = sym_family_fit(X, 60, 2, y, 0, 0, &p);
    CHECK(m);
    CHECK(sym_family_predict(m, X, 60, 2, 0, out) == 0);
    double mse = 0;
    for (int i = 0; i < 60; i++)
        mse += (out[i] - y[i]) * (out[i] - y[i]);
    CHECK(mse / 60 < .1);
    sym_family_model_t *again = sym_family_fit(X, 60, 2, y, 0, 0, &p);
    CHECK(again);
    double a[6 * SYM_FAMILY_MAX_TERMS + 6], b[6 * SYM_FAMILY_MAX_TERMS + 6];
    int n = sym_family_export(m, 0, -1, a, sizeof(a) / sizeof(double));
    CHECK(n == 6 * p.terms + 6 && sym_family_export(again, 0, -1, b, n) == n);
    CHECK(a[n - 3] == 7);
    for (int t = 0; t < p.terms; t++) {
        CHECK(a[6 * t + 3] == 1);
        CHECK(a[6 * t + 4] == 0);
    }
    CHECK(memcmp(a, b, n * sizeof(double)) == 0);
    sym_family_model_t *loaded = sym_family_new(2, 0, p.terms, p.frontier);
    CHECK(loaded && sym_family_import(loaded, 0, -1, a, n) == 0);
    CHECK(sym_family_predict(loaded, X, 60, 2, 0, out) == 0);
    a[0] = NAN;
    CHECK(sym_family_import(loaded, 0, -1, a, n) == -1);
    CHECK(sym_family_predict(loaded, X, 60, 2, 0, out) == 0);
    /* A rejected or malformed gradient proposal must preserve the artifact. */
    double updates[3 * SYM_FAMILY_MAX_TERMS + 1], report[6], split[121];
    CHECK(sym_family_export(m, 0, -1, a, n) == n);
    for (int t = 0; t < p.terms; t++) {
        updates[3 * t] = a[6 * t + 3];
        updates[3 * t + 1] = a[6 * t + 4];
        updates[3 * t + 2] = a[6 * t + 5];
    }
    updates[3 * p.terms] = 1e10;
    CHECK(sym_family_refine_accept(m, 0, updates, 3 * p.terms + 1, X, 60, 2, y, .2, p.seed, 0,
                                   report) == 0);
    CHECK(report[5] > report[2]);
    CHECK(sym_family_export(m, 0, -1, b, n) == n && memcmp(a, b, n * sizeof(double)) == 0);
    updates[0] = NAN;
    CHECK(sym_family_refine_accept(m, 0, updates, 3 * p.terms + 1, X, 60, 2, y, .2, p.seed, 0,
                                   report) == -1);
    CHECK(sym_family_export(m, 0, -1, b, n) == n && memcmp(a, b, n * sizeof(double)) == 0);
    CHECK(sym_family_refine_data(m, 0, y, 60, .2, p.seed, split, 121) == 0);
    double changed[60], split2[121];
    for (int r = 0; r < 60; r++)
        changed[r] = split[60 + r] ? y[r] : y[r] + 100;
    CHECK(sym_family_refine_data(m, 0, changed, 60, .2, p.seed, split2, 121) == 0);
    CHECK(split[120] == split2[120]);
    for (int r = 0; r < 60; r++) {
        CHECK(split[60 + r] == split2[60 + r]);
        if (split[60 + r])
            CHECK(split[r] == split2[r]);
    }
    sym_family_free(loaded);
    sym_family_free(again);
    sym_family_free(m);
    for (int k = 2; k <= 3; k++) {
        for (int i = 0; i < 60; i++)
            y[i] = (double)(i % k);
        m = sym_family_fit(X, 60, 2, y, 1, k, &p);
        CHECK(m);
        CHECK(sym_family_predict(m, X, 60, 2, 2, out) == 0);
        for (int i = 0; i < 60; i++) {
            double sum = 0;
            for (int j = 0; j < k; j++) {
                CHECK(out[i * k + j] >= 0 && out[i * k + j] <= 1);
                sum += out[i * k + j];
            }
            CHECK(fabs(sum - 1) < 1e-12);
        }
        sym_family_free(m);
    }
    p.terms = 0;
    CHECK(sym_family_fit(X, 60, 2, y, 0, 0, &p) == NULL);
    puts("family: deterministic fit, prediction, import validation and class probabilities passed");
    return 0;
}

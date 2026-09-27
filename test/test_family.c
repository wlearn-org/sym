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
int main(void) {
    double X[120], y[60], out[180];
    for (int i = 0; i < 60; i++) {
        X[2 * i] = (i - 30) / 10.0;
        X[2 * i + 1] = sin(i * .47);
        y[i] = 2 * X[2 * i] + .3 * X[2 * i + 1] + 1;
    }
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

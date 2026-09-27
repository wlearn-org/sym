/* Capture/compare the complete raw artifact and predictions against a separately
 * built pre-refactor library. Compile this same source against each library. */
#include "sym.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    enum { N = 48, D = 3 };
    double X[N * D], y[N], pred[N * 4];
    int cases = 0;
    for (int task = 0; task < 6; task++) {
        for (int profile = 0; profile < 4; profile++) {
            for (int seed = 1; seed <= 2; seed++) {
                for (int i = 0; i < N; i++) {
                    double a = (i - 23.0) / 13.0;
                    X[i * D] = a;
                    X[i * D + 1] = sin(i * 0.73);
                    X[i * D + 2] = cos(i * 0.39);
                    y[i] = task == 3 ? (double)(i % 2) : task == 4 ? (double)(i % 3)
                        : a * a + sin(a) + 0.17 * X[i * D + 1];
                }
                sym_params_t p;
                sym_params_init(&p);
                p.task = task < 3 ? SYM_TASK_REGRESSION : task < 5 ? SYM_TASK_CLASSIFICATION : SYM_TASK_TRANSFORMER;
                p.loss = task < 3 ? task : SYM_LOSS_MSE;
                p.n_classes = task == 4 ? 3 : 2;
                p.population = 36;
                p.generations = 6;
                p.frontier_size = 6;
                p.top_k = 3;
                p.elite_count = 4;
                p.max_nodes = 15;
                p.max_depth = 4;
                p.seed = seed * 37;
                p.operator_set = profile % 3;
                p.validation_fraction = profile == 0 ? 0 : 0.25;
                p.early_stop_rounds = profile == 3 ? 2 : 0;
                if (profile > 0) {
                    p.islands = 3;
                    p.migration_interval = 2;
                    p.migration_count = 1;
                    p.brood_size = 3;
                    p.warmup_generations = 3;
                    p.warmup_min_nodes = 7;
                }
                if (profile > 1) {
                    p.row_sample_size = 20;
                    p.local_refine_count = 2;
                    p.local_refine_interval = 2;
                    p.complexity_hof_size = 8;
                    p.final_selector = profile == 2 ? SYM_FINAL_LOSS : SYM_FINAL_SCORE;
                }
                sym_model_t *m = sym_fit(X, N, D, y, &p);
                if (!m) { fprintf(stderr, "%s\n", sym_get_error()); return 1; }
                char *raw = NULL;
                int32_t len = 0;
                if (sym_save(m, &raw, &len) || sym_predict_raw(m, X, N, D, pred)) return 1;
                char path[4096];
                snprintf(path, sizeof(path), "%s/case-%d-%d-%d.bin", argv[1], task, profile, seed);
                FILE *f = fopen(path, "wb");
                if (!f) return 1;
                size_t n = (size_t)N * m->n_outputs;
                int ok = fwrite(&len, sizeof(len), 1, f) == 1 &&
                    fwrite(raw, 1, len, f) == (size_t)len &&
                    fwrite(pred, sizeof(double), n, f) == n;
                if (fclose(f) || !ok) return 1;
                sym_free_buffer(raw);
                sym_free(m);
                cases++;
            }
        }
    }
    printf("Captured %d tree cases (raw models and predictions)\n", cases);
    return 0;
}

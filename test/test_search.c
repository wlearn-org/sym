#include "sym.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "line %d: %s (%s)\n", __LINE__, #c, sym_get_error()); return 1; } } while (0)

int main(void) {
    double X[96], y[48];
    for (int i = 0; i < 48; i++) {
        X[2*i] = sin(i * 0.37); X[2*i+1] = i / 19.0;
        y[i] = X[2*i] * X[2*i+1] + 0.2;
    }
    int cases = 0;
    for (int task = 0; task < 3; task++) {
        for (int cap = 1; cap <= 64; cap *= 4) {
            sym_params_t p; sym_params_init(&p);
            p.task = task; p.n_classes = 3; p.top_k = 3;
            p.population = 24; p.generations = 4; p.max_nodes = 15;
            p.frontier_size = 5; p.elite_count = 3; p.seed = 12;
            p.islands = 3; p.migration_interval = 2; p.migration_count = 1;
            p.brood_size = 3; p.row_sample_size = 20;
            p.local_refine_count = 2; p.local_refine_interval = 2;
            p.complexity_hof_size = 5;
            double labels[48];
            for (int i = 0; i < 48; i++) labels[i] = task == 1 ? i % 3 : y[i];
            sym_model_t *direct = sym_fit(X, 48, 2, labels, &p);
            CHECK(direct);
            sym_search_t *s = sym_search_new(X, 48, 2, labels, &p, cap);
            CHECK(s);
            CHECK(sym_search_finish(s) == NULL);
            const sym_batch_t *b;
            sym_search_score_t scores[64];
            int rc, batches = 0;
            while ((rc = sym_search_propose(s, &b)) > 0) {
                CHECK(b->count > 0 && b->count <= cap);
                uint32_t id = b->id;
                const sym_batch_t *again;
                CHECK(sym_search_propose(s, &again) == 1 && again == b && again->id == id);
                CHECK(sym_search_score(s, id, scores, b->count) == 0);
                CHECK(sym_search_accept(s, id + 1, scores, b->count) == -1);
                CHECK(sym_search_accept(s, id, scores, b->count - 1) == -1);
                double loss = scores[0].train_loss;
                scores[0].train_loss = NAN;
                CHECK(sym_search_accept(s, id, scores, b->count) == -1);
                scores[0].train_loss = loss;
                if (b->count > 1) {
                    double original = b->formulas[0]->train_loss;
                    double second = scores[1].train_loss;
                    scores[0].train_loss = 123.0;
                    scores[1].train_loss = NAN;
                    CHECK(sym_search_accept(s, id, scores, b->count) == -1);
                    CHECK(b->formulas[0]->train_loss == original);
                    scores[0].train_loss = loss;
                    scores[1].train_loss = second;
                }
                CHECK(sym_search_accept(s, id, scores, b->count) == 0);
                CHECK(sym_search_accept(s, id, scores, b->count) == -1);
                batches++;
            }
            CHECK(rc == 0 && batches > 1);
            sym_model_t *stepped = sym_search_finish(s);
            CHECK(stepped);
            CHECK(sym_search_finish(s) == NULL);
            sym_search_free(s);
            char *a, *braw; int32_t na, nb;
            CHECK(sym_save(direct, &a, &na) == 0);
            CHECK(sym_save(stepped, &braw, &nb) == 0);
            CHECK(na == nb && memcmp(a, braw, na) == 0);
            sym_free_buffer(a); sym_free_buffer(braw);
            sym_free(direct); sym_free(stepped);
            /* Cancellation with an outstanding batch must release its formulas. */
            s = sym_search_new(X, 48, 2, labels, &p, cap);
            CHECK(s && sym_search_propose(s, &b) == 1);
            sym_search_free(s);
            cases++;
        }
    }
    printf("%d stepped-search parity/lifecycle cases passed\n", cases);
    return 0;
}

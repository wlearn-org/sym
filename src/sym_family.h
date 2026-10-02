#ifndef WLEARN_SYM_FAMILY_H
#define WLEARN_SYM_FAMILY_H
#include "sym.h"
#ifdef __cplusplus
extern "C" {
#endif
#define SYM_FAMILY_MAX_TERMS 32

typedef struct {
    int32_t population, generations, terms, elite, islands, frontier;
    uint32_t operators, seed;
    int32_t polish, patience;
    double validation_fraction, complexity_penalty, immigrant_rate;
    double mutation_rate, crossover_rate, ridge, tol;
    /* 0: sequential coordinate polish; 1..512: frozen-base variants per round. */
    int32_t polish_batch;
    int32_t hierarchical;
} sym_family_params_t;
typedef struct sym_family_model sym_family_model_t;
void sym_family_params_init(sym_family_params_t *p);
/* Fit borrows input buffers only for the call and returns an owned model.
 * task 0: regression, classes=0. Task 1: encoded labels [0,classes),
 * classes>=2. Free every returned model with sym_family_free(). */
sym_family_model_t *sym_family_fit(const double *X, int32_t rows, int32_t cols, const double *y,
                                   int32_t task, int32_t classes, const sym_family_params_t *p);
/* Optional in-loop polish; preserves the original parameter-struct ABI. Both
 * controls zero disables it; otherwise both must be positive, count <= 32 and
 * population, with polish > 0. Reuses polish/polish_batch before breeding. */
sym_family_model_t *sym_family_fit_refined(const double *X, int32_t rows, int32_t cols,
                                           const double *y, int32_t task, int32_t classes,
                                           const sym_family_params_t *p, int32_t interval,
                                           int32_t count);
/* Feature-tile protocol: descriptors are [candidate,term,5] doubles containing
 * featureA,featureB,operator,p0,p1. Evaluators return [candidate,row,term] doubles.
 * stage 0 accumulates train-only moments, 1 updates QR, 2 computes losses.
 * X/descriptors are borrowed until accept/free; no view may cross an async or
 * Wasm memory-growth boundary. new copies inputs. Invalid accept is atomic.
 * Sequential polish proposes one dependent trial; batched polish freezes the
 * base for a round. No Gram solve or pre-ranking. */
typedef struct sym_family_search sym_family_search_t;
typedef struct {
    uint32_t id;
    int32_t stage, head, generation, candidates, rows, cols, terms, row_start;
    const double *X, *descriptors;
} sym_family_batch_t;
sym_family_search_t *sym_family_search_new(const double *X, int32_t rows, int32_t cols,
                                           const double *y, int32_t task, int32_t classes,
                                           const sym_family_params_t *p, int32_t capacity,
                                           int32_t tile_rows);
/* Configure only before the first proposal; invalid changes leave state intact. */
int sym_family_search_set_refinement(sym_family_search_t *s, int32_t interval, int32_t count);
int sym_family_search_propose(sym_family_search_t *s, const sym_family_batch_t **out);
const sym_family_batch_t *sym_family_search_pending(const sym_family_search_t *s);
int sym_family_search_score(const sym_family_search_t *s, uint32_t id, double *out, int32_t count);
int sym_family_search_accept(sym_family_search_t *s, uint32_t id, const double *data,
                             int32_t count);
/* Reduced evaluator protocol, alternative to the three feature-tile passes.
 * Only a fresh pending stage-0/row-0 batch may be solved/accepted. Data packs
 * [targets(rows), training_mask(rows), training_target_mean]. Descriptors and
 * target data must be copied before yielding. Factors are [candidate,width,width]
 * QR R factors of centered/scaled [features,target]; width >= terms+1.
 * Stats are [candidate,3*terms]: anchor, shifted mean, positive scale. C applies
 * original-unit ridge once and rounds coefficients exactly as the reference.
 * Results pack [terms coefficients,bias,trainMSE,validMSE] per candidate.
 * All lengths are in doubles. Malformed acceptance leaves the batch pending. */
int sym_family_search_data(const sym_family_search_t *s, double *out, int32_t count);
int sym_family_search_solve(const sym_family_search_t *s, uint32_t id, const double *factors,
                            int32_t factor_count, const double *stats, int32_t stats_count,
                            int32_t width, double *coefficients, int32_t coefficient_count);
int sym_family_search_accept_results(sym_family_search_t *s, uint32_t id, const double *results,
                                     int32_t count);
uint64_t sym_family_search_evaluations(const sym_family_search_t *s);
sym_family_model_t *sym_family_search_finish(sym_family_search_t *s);
void sym_family_search_free(sym_family_search_t *s);

/* Model construction/import is for the existing family payload, not SYM1 trees.
 * Index -1 is the selected formula; other indices address the head's archive.
 * Packed doubles: [featureA,featureB,op,p0,p1,coefficient] repeated terms times,
 * then [bias,trainMSE,validMSE,complexity,objective,generation].
 * op 0..9: add,sub,mul,div,sin,cos,tanh,logabs,sqrtabs,expclamp.
 * Import validates before replacing state; both operations copy the buffer. */
sym_family_model_t *sym_family_new(int32_t cols, int32_t classes, int32_t terms, int32_t frontier);
int sym_family_import(sym_family_model_t *m, int32_t head, int32_t index, const double *data,
                      int32_t count);
int sym_family_export(const sym_family_model_t *m, int32_t head, int32_t index, double *data,
                      int32_t count);
int sym_family_archive_size(const sym_family_model_t *m, int32_t head);
/* mode: 0 predictions/encoded class indices (rows outputs),
 * 1 raw margins (rows*heads outputs), 2 probabilities (rows*classes outputs).
 * heads is classes for multiclass, otherwise 1. Buffers are caller-owned. */
int sym_family_predict(const sym_family_model_t *m, const double *X, int32_t rows, int32_t cols,
                       int32_t mode, double *out);
/* SYM2 family payload, kind=1, semantics=2 (flat) or 3 (hierarchical).
 * Hierarchical source indices cols+t reference earlier terms only.
 * Save allocates bytes freed with sym_free_buffer; load validates exact lengths
 * before allocation. Legacy imports remain @1 and cannot be silently upgraded. */
int sym_family_save(const sym_family_model_t *m, char **out, int32_t *length);
sym_family_model_t *sym_family_load(const char *data, int32_t length);
int sym_family_dimensions(const sym_family_model_t *m, int32_t out[4]);
/* Gradient refinement uses the search split and C evaluation for acceptance.
 * data packs targets, training mask, mean as above. Updates contain p0,p1,coef
 * per term then bias. Structural choices never change. report[6] contains
 * before train/valid/objective then proposed train/valid/objective.
 * Returns 1 committed, 0 rejected without mutation, -1 invalid proposal. */
int sym_family_refine_data(const sym_family_model_t *, int32_t head, const double *y, int32_t rows,
                           double fraction, uint32_t seed, double *out, int32_t count);
int sym_family_refine_accept(sym_family_model_t *, int32_t head, const double *updates,
                             int32_t count, const double *X, int32_t rows, int32_t cols,
                             const double *y, double fraction, uint32_t seed, double tolerance,
                             double *report);
void sym_family_free(sym_family_model_t *m);
#ifdef __cplusplus
}
#endif
#endif

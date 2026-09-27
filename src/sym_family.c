#include "sym_family.h"
#include "sym_family_internal.h"
#include "sym_internal.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Search mechanics adapted from symcpg's C family experiment; the grammar and
 * float32 term/readout storage preserve sym's existing family JSON semantics.
 * No benchmark datasets, compound operators or Polygrad internals enter here. */
static int fail(const char *s) {
    sym_set_error(s);
    return -1;
}
static double clamp(double x, double lo, double hi) { return x < lo ? lo : x > hi ? hi : x; }
static double f32(double x) { return (double)(float)x; }
static uint32_t next_u32(uint32_t *r) {
    *r = 1664525u * *r + 1013904223u;
    return *r;
}
static double uniform(uint32_t *r) { return next_u32(r) / 4294967296.0; }
static int randint(uint32_t *r, int n) { return (int)(uniform(r) * n); }
static double signed_uniform(uint32_t *r) { return 2 * uniform(r) - 1; }
static double term(int op, double a, double b, double p0, double p1) {
    double z = a * p0 + p1;
    switch (op) {
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
    case 9:
        return exp(clamp(z, -6, 6));
    default:
        return NAN;
    }
}
static int features(const candidate *c, const double *x, int terms, double *out) {
    for (int t = 0; t < terms; t++) {
        out[t] = term(c->op[t], f32(x[c->a[t]]), f32(x[c->b[t]]), c->p0[t], c->p1[t]);
        if (!isfinite(out[t]))
            return -1;
    }
    return 0;
}
static double prediction(const candidate *c, const double *phi, int terms) {
    double p = c->bias;
    for (int t = 0; t < terms; t++)
        p += c->coef[t] * phi[t];
    return p;
}
void sym_family_params_init(sym_family_params_t *p) {
    if (!p)
        return;
    memset(p, 0, sizeof(*p));
    p->population = 128;
    p->generations = 20;
    p->terms = 6;
    p->elite = 8;
    p->islands = 1;
    p->frontier = 16;
    p->operators = 1023;
    p->seed = 42;
    p->complexity_penalty = 1e-5;
    p->immigrant_rate = 0;
    p->mutation_rate = .35;
    p->crossover_rate = .55;
    p->ridge = 1e-8;
    p->tol = 1e-12;
}
static int validate_params(const sym_family_params_t *p) {
    if (!p || p->population < 4 || p->population > 4096 || p->generations < 1 ||
        p->generations > 100000 || p->terms < 1 || p->terms > SYM_FAMILY_MAX_TERMS ||
        p->elite < 1 || p->elite >= p->population || p->islands < 1 ||
        p->islands > p->population / 2 || p->frontier < 1 || p->frontier > 128 ||
        !p->operators || (p->operators & ~1023u) || p->polish < 0 || p->polish > 40 ||
        p->patience < 0 || !isfinite(p->validation_fraction) || p->validation_fraction < 0 ||
        p->validation_fraction >= 1 || !isfinite(p->complexity_penalty) ||
        p->complexity_penalty < 0 || !isfinite(p->immigrant_rate) || p->immigrant_rate < 0 ||
        p->immigrant_rate > .5 || !isfinite(p->mutation_rate) || p->mutation_rate < 0 ||
        p->mutation_rate > 1 || !isfinite(p->crossover_rate) || p->crossover_rate < 0 ||
        p->crossover_rate > 1 || !isfinite(p->ridge) || p->ridge < 0 || !isfinite(p->tol) ||
        p->tol < 0)
        return fail("invalid family search parameters");
    return 0;
}
sym_family_model_t *sym_family_new(int32_t cols, int32_t classes, int32_t terms,
                                   int32_t frontier) {
    if (cols < 1 || classes < 0 || classes == 1 || classes > 128 || terms < 1 ||
        terms > SYM_FAMILY_MAX_TERMS || frontier < 1 || frontier > 128) {
        fail("invalid family model dimensions");
        return NULL;
    }
    sym_family_model_t *m = calloc(1, sizeof(*m));
    if (!m) {
        fail("out of memory allocating family model");
        return NULL;
    }
    m->cols = cols;
    m->classes = classes;
    m->heads = classes > 2 ? classes : 1;
    m->terms = terms;
    m->frontier = frontier;
    m->semantics = 1;
    m->best = calloc((size_t)m->heads, sizeof(candidate));
    m->archive = calloc((size_t)m->heads * frontier, sizeof(candidate));
    m->counts = calloc((size_t)m->heads, sizeof(int32_t));
    m->fitted = calloc((size_t)m->heads, 1);
    if (!m->best || !m->archive || !m->counts || !m->fitted) {
        fail("out of memory allocating family formulas");
        sym_family_free(m);
        return NULL;
    }
    return m;
}
void sym_family_free(sym_family_model_t *m) {
    if (!m)
        return;
    free(m->best);
    free(m->archive);
    free(m->counts);
    free(m->fitted);
    free(m);
}
static int compare(const void *ap, const void *bp) {
    const candidate *a = ap, *b = bp;
    if (a->objective != b->objective)
        return a->objective < b->objective ? -1 : 1;
    if (a->valid != b->valid)
        return a->valid < b->valid ? -1 : 1;
    if (a->complexity != b->complexity)
        return a->complexity < b->complexity ? -1 : 1;
    return (a->ordinal > b->ordinal) - (a->ordinal < b->ordinal);
}
static int same_genes(const candidate *a, const candidate *b, int n) {
    for (int t = 0; t < n; t++)
        if (a->a[t] != b->a[t] || a->b[t] != b->b[t] || a->op[t] != b->op[t] ||
            a->p0[t] != b->p0[t] || a->p1[t] != b->p1[t])
            return 0;
    return 1;
}
static void archive_candidate(sym_family_model_t *m, int h, const candidate *c) {
    if (c->objective == DBL_MAX)
        return;
    candidate *a = m->archive + (size_t)h * m->frontier;
    int n = m->counts[h], at = n;
    for (int i = 0; i < n; i++)
        if (same_genes(c, &a[i], m->terms)) {
            if (compare(c, &a[i]) >= 0)
                return;
            at = i;
            break;
        }
    if (at == n && n == m->frontier) {
        if (compare(c, &a[n - 1]) >= 0)
            return;
        at = n - 1;
    }
    a[at] = *c;
    if (at == n)
        m->counts[h] = ++n;
    qsort(a, (size_t)n, sizeof(candidate), compare);
}
/* Unused genes have one representation. Their values cannot affect complexity,
 * mutation or archive identity. Legacy metrics are retained only on @1 import. */
static void canonicalize(candidate *c, int terms) {
    for (int t = 0; t < terms; t++) {
        if (c->op[t] < 4) {
            c->p0[t] = 1;
            c->p1[t] = 0;
        } else {
            c->b[t] = 0;
        }
    }
}
double sym_family_complexity(const candidate *c, int terms) {
    double value = 1;
    for (int t = 0; t < terms; t++) {
        value += c->op[t] < 4 ? 2 : 3;
        if (c->op[t] >= 4) {
            value += fabs(c->p0[t] - 1) > 1e-6;
            value += fabs(c->p1[t]) > 1e-6;
        }
    }
    return value;
}

static int random_op(uint32_t *rng, uint32_t mask) {
    int ids[10], count = 0;
    for (int i = 0; i < 10; i++)
        if (mask & (1u << i))
            ids[count++] = i;
    return ids[randint(rng, count)];
}
static void init_candidate(candidate *c, uint32_t *rng, int cols,
                           const sym_family_params_t *p) {
    memset(c, 0, sizeof(*c));
    for (int t = 0; t < p->terms; t++) {
        c->a[t] = randint(rng, cols);
        c->b[t] = randint(rng, cols);
        c->op[t] = random_op(rng, p->operators);
        c->p0[t] = f32(4 * signed_uniform(rng));
        c->p1[t] = f32(4 * signed_uniform(rng));
    }
    canonicalize(c, p->terms);
}
/* Adapted symcpg mutation schedule; coefficients are refit, not mutated. */
static void mutate(candidate *c, uint32_t *rng, int cols, int gen,
                   const sym_family_params_t *p) {
    double scale = .08 + .92 * (1 - (double)gen / p->generations);
    for (int t = 0; t < p->terms; t++) {
        if (uniform(rng) < p->mutation_rate)
            c->a[t] = randint(rng, cols);
        if (uniform(rng) < p->mutation_rate)
            c->b[t] = randint(rng, cols);
        if (uniform(rng) < p->mutation_rate)
            c->op[t] = random_op(rng, p->operators);
        c->p0[t] = f32(clamp(c->p0[t] + signed_uniform(rng) * 1.5 * scale, -16, 16));
        c->p1[t] = f32(clamp(c->p1[t] + signed_uniform(rng) * 1.5 * scale, -16, 16));
    }
    canonicalize(c, p->terms);
    c->ready = 0;
}
static void evolve(candidate *pop, candidate *next, uint32_t *rng, int cols, int gen,
                   const sym_family_params_t *p) {
    for (int island = 0; island < p->islands; island++) {
        int start = island * p->population / p->islands,
            end = (island + 1) * p->population / p->islands;
        int size = end - start, elite = (p->elite + p->islands - 1) / p->islands;
        if (elite > size / 2)
            elite = size / 2;
        if (elite < 1)
            elite = 1;
        qsort(pop + start, (size_t)size, sizeof(candidate), compare);
        int immigrants = (int)(size * p->immigrant_rate);
        int pool = size / 2;
        for (int i = start; i < end; i++) {
            if (i < start + elite)
                next[i] = pop[i];
            else if (i >= end - immigrants)
                init_candidate(&next[i], rng, cols, p);
            else {
                next[i] = pop[start + randint(rng, pool)];
                if (uniform(rng) < p->crossover_rate) {
                    const candidate *other = &pop[start + randint(rng, pool)];
                    for (int t = 0; t < p->terms; t++)
                        if (uniform(rng) < .5) {
                            next[i].a[t] = other->a[t];
                            next[i].b[t] = other->b[t];
                            next[i].op[t] = other->op[t];
                            next[i].p0[t] = other->p0[t];
                            next[i].p1[t] = other->p1[t];
                        }
                }
                mutate(&next[i], rng, cols, gen, p);
            }
            next[i].ordinal = i;
        }
    }
}
/* The evaluator supplies feature tiles, not normal-equation summaries. C owns
 * centering, ridge QR and loss reductions for every backend. Repeating features
 * over three ordered passes bounds memory without changing the solver. */
typedef struct {
    double mean[SYM_FAMILY_MAX_TERMS], moment[SYM_FAMILY_MAX_TERMS];
    double scale[SYM_FAMILY_MAX_TERMS], R[SYM_FAMILY_MAX_TERMS * SYM_FAMILY_MAX_TERMS];
    double z[SYM_FAMILY_MAX_TERMS], ym, train, valid;
    int n, nv, invalid;
} family_qr;

enum family_phase { FAMILY_POPULATION, FAMILY_POLISH, FAMILY_DONE, FAMILY_TAKEN };
struct sym_family_search {
    sym_family_model_t *model;
    sym_family_params_t params;
    double *X, *y, *descriptors;
    uint8_t *validation;
    candidate *population, *offspring, *working;
    candidate selected, trial;
    family_qr *qr;
    int32_t rows, cols, capacity, tile_rows, head, generation, cursor, count;
    int32_t slots[64], pass, row, pending, stale;
    int32_t polish_pass, polish_term, polish_param, polish_sign, polish_improved;
    double best, polish_step;
    uint32_t rng, serial;
    uint64_t evaluations;
    enum family_phase phase;
    sym_family_batch_t batch;
};

static double family_target(const sym_family_search_t *s, int row) {
    if (!s->model->classes)
        return s->y[row];
    int positive = s->model->classes == 2 ? 1 : s->head;
    return s->y[row] == positive ? 1 : -1;
}

/* Unpenalized intercept; ridge is in original feature units, with objective
 * MSE + ridge*||coef||^2. Only training rows enter means, scales and QR. */
static void qr_row(family_qr *q, const double *input, double target, int terms) {
    double phi[SYM_FAMILY_MAX_TERMS];
    for (int j = 0; j < terms; j++)
        phi[j] = (input[j] - q->mean[j]) / q->scale[j];
    double y = target - q->ym;
    for (int j = 0; j < terms; j++) {
        double h = hypot(q->R[j * terms + j], phi[j]);
        if (h == 0)
            continue;
        double co = q->R[j * terms + j] / h, si = phi[j] / h;
        for (int k = j; k < terms; k++) {
            double old = q->R[j * terms + k];
            q->R[j * terms + k] = co * old + si * phi[k];
            phi[k] = -si * old + co * phi[k];
        }
        double old = q->z[j];
        q->z[j] = co * old + si * y;
        y = -si * old + co * y;
    }
}
static void qr_solve(family_qr *q, candidate *c, int terms) {
    double beta[SYM_FAMILY_MAX_TERMS] = {0};
    c->bias = q->ym;
    for (int j = terms - 1; j >= 0; j--) {
        double v = q->z[j];
        for (int k = j + 1; k < terms; k++)
            v -= q->R[j * terms + k] * beta[k];
        beta[j] = fabs(q->R[j * terms + j]) <= DBL_EPSILON ? 0 : v / q->R[j * terms + j];
        c->coef[j] = f32(beta[j] / q->scale[j]);
        if (!isfinite(c->coef[j]))
            q->invalid = 1;
        c->bias -= q->mean[j] * c->coef[j];
    }
    if (!isfinite(c->bias))
        q->invalid = 1;
}
static void family_start_head(sym_family_search_t *s) {
    s->rng = s->params.seed + (uint32_t)s->head * 104729u;
    for (int i = 0; i < s->params.population; i++) {
        init_candidate(&s->population[i], &s->rng, s->cols, &s->params);
        s->population[i].ordinal = i;
    }
    s->generation = s->cursor = s->stale = 0;
    s->best = DBL_MAX;
    s->phase = FAMILY_POPULATION;
}
void sym_family_search_free(sym_family_search_t *s) {
    if (!s)
        return;
    sym_family_free(s->model);
    free(s->X);
    free(s->y);
    free(s->validation);
    free(s->population);
    free(s->offspring);
    free(s->working);
    free(s->qr);
    free(s->descriptors);
    free(s);
}
sym_family_search_t *sym_family_search_new(const double *X, int32_t rows, int32_t cols,
                                           const double *y, int32_t task, int32_t classes,
                                           const sym_family_params_t *p, int32_t capacity,
                                           int32_t tile_rows) {
    if (validate_params(p))
        return NULL;
    if (!X || !y || rows < 2 || cols < 1 || (task != 0 && task != 1) ||
        (task == 0 && classes != 0) || (task == 1 && (classes < 2 || classes > 128)) ||
        capacity < 1 || capacity > 64 || tile_rows < 1 || tile_rows > 4096 ||
        (size_t)rows > SIZE_MAX / sizeof(double) / (size_t)cols) {
        fail("invalid family training shape, task or tile capacity");
        return NULL;
    }
    for (size_t i = 0; i < (size_t)rows * cols; i++) {
        if (!isfinite(f32(X[i]))) {
            fail("family inputs must be finite float32 values");
            return NULL;
        }
    }
    for (int i = 0; i < rows; i++) {
        if (!isfinite(f32(y[i])) ||
            (task && (y[i] < 0 || y[i] >= classes || floor(y[i]) != y[i]))) {
            fail("invalid family target");
            return NULL;
        }
    }
    int nv = (int)floor(rows * p->validation_fraction);
    if (p->validation_fraction > 0 && nv == 0)
        nv = 1;
    if (rows - nv < 2) {
        fail("family fitting requires at least two training rows");
        return NULL;
    }
    sym_family_search_t *s = calloc(1, sizeof(*s));
    if (!s) {
        fail("out of memory allocating family search");
        return NULL;
    }
    s->params = *p;
    s->rows = rows;
    s->cols = cols;
    s->capacity = capacity;
    s->tile_rows = tile_rows;
    s->model = sym_family_new(cols, classes, p->terms, p->frontier);
    s->X = malloc((size_t)rows * cols * sizeof(double));
    s->y = malloc((size_t)rows * sizeof(double));
    s->validation = calloc((size_t)rows, 1);
    s->population = calloc((size_t)p->population, sizeof(candidate));
    s->offspring = calloc((size_t)p->population, sizeof(candidate));
    s->working = calloc((size_t)capacity, sizeof(candidate));
    s->qr = calloc((size_t)capacity, sizeof(family_qr));
    s->descriptors = calloc((size_t)capacity * p->terms * 5, sizeof(double));
    int32_t *order = malloc((size_t)rows * sizeof(int32_t));
    if (!s->model || !s->X || !s->y || !s->validation || !s->population || !s->offspring ||
        !s->working || !s->qr || !s->descriptors || !order) {
        free(order);
        sym_family_search_free(s);
        fail("out of memory in family search");
        return NULL;
    }
    s->model->semantics = 2;
    s->model->penalty = p->complexity_penalty;
    for (size_t i = 0; i < (size_t)rows * cols; i++)
        s->X[i] = f32(X[i]);
    for (int i = 0; i < rows; i++) {
        s->y[i] = f32(y[i]);
        order[i] = i;
    }
    uint32_t split = p->seed ^ 0x9e3779b9u;
    for (int i = 0; i < nv; i++) {
        int j = i + randint(&split, rows - i), tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
        s->validation[order[i]] = 1;
    }
    free(order);
    family_start_head(s);
    return s;
}
static int family_begin_work(sym_family_search_t *s) {
    s->row = s->pass = 0;
    memset(s->qr, 0, (size_t)s->count * sizeof(family_qr));
    for (int i = 0; i < s->count; i++) {
        candidate *c = &s->working[i];
        c->train = c->valid = c->objective = DBL_MAX;
        for (int t = 0; t < s->params.terms; t++) {
            double *d = s->descriptors + ((size_t)i * s->params.terms + t) * 5;
            d[0] = c->a[t];
            d[1] = c->b[t];
            d[2] = c->op[t];
            d[3] = c->p0[t];
            d[4] = c->p1[t];
        }
    }
    return 0;
}
static void family_finish_head(sym_family_search_t *s) {
    archive_candidate(s->model, s->head, &s->selected);
    s->model->best[s->head] = s->model->archive[(size_t)s->head * s->params.frontier];
    s->model->fitted[s->head] = 1;
    if (++s->head == s->model->heads)
        s->phase = FAMILY_DONE;
    else
        family_start_head(s);
}
static int family_next_work(sym_family_search_t *s) {
    while (s->phase != FAMILY_DONE) {
        if (s->phase == FAMILY_POPULATION) {
            s->count = 0;
            while (s->cursor < s->params.population && s->count < s->capacity) {
                int at = s->cursor++;
                // Elites keep their fit: data, mask and objective never change.
                if (s->population[at].ready) {
                    s->population[at].generation = s->generation;
                    archive_candidate(s->model, s->head, &s->population[at]);
                    continue;
                }
                s->slots[s->count] = at;
                s->working[s->count] = s->population[at];
                s->working[s->count++].generation = s->generation;
            }
            if (s->count)
                return family_begin_work(s) + 1;
            if (!s->model->counts[s->head])
                return fail(
                    "no finite family candidate; rescale input data or change operators");
            double loss = s->model->archive[(size_t)s->head * s->params.frontier].objective;
            if (loss + s->params.tol < s->best) {
                s->best = loss;
                s->stale = 0;
            } else
                s->stale++;
            if (s->generation + 1 == s->params.generations ||
                (s->params.patience > 0 && s->stale >= s->params.patience)) {
                s->selected = s->model->archive[(size_t)s->head * s->params.frontier];
                s->phase = FAMILY_POLISH;
                s->polish_pass = s->polish_term = s->polish_param = s->polish_improved = 0;
                s->polish_sign = -1;
                s->polish_step = .75;
            } else {
                evolve(s->population, s->offspring, &s->rng, s->cols, s->generation,
                       &s->params);
                candidate *tmp = s->population;
                s->population = s->offspring;
                s->offspring = tmp;
                s->generation++;
                s->cursor = 0;
            }
        } else if (s->phase == FAMILY_POLISH) {
            while (s->polish_pass < s->params.polish) {
                while (s->polish_term < s->params.terms && s->selected.op[s->polish_term] < 4)
                    s->polish_term++;
                if (s->polish_term == s->params.terms) {
                    if (!s->polish_improved)
                        s->polish_step *= .55;
                    s->polish_pass++;
                    s->polish_term = s->polish_param = s->polish_improved = 0;
                    s->polish_sign = -1;
                    continue;
                }
                // One dependent coordinate trial per work unit, never a batch
                // of trials built from an already obsolete selected candidate.
                s->trial = s->selected;
                double *value = s->polish_param ? &s->trial.p1[s->polish_term]
                                                : &s->trial.p0[s->polish_term];
                *value = f32(clamp(*value + s->polish_sign * s->polish_step, -16, 16));
                s->working[0] = s->trial;
                s->count = 1;
                return family_begin_work(s) + 1;
            }
            family_finish_head(s);
        }
    }
    return 0;
}
int sym_family_search_propose(sym_family_search_t *s, const sym_family_batch_t **out) {
    if (!s || !out || s->phase == FAMILY_TAKEN)
        return fail("invalid family search state");
    *out = NULL;
    if (s->pending) {
        *out = &s->batch;
        return 1;
    }
    if (!s->count) {
        int rc = family_next_work(s);
        if (rc <= 0)
            return rc;
    }
    if (s->serial == UINT32_MAX)
        return fail("family batch identity exhausted");
    sym_family_batch_t *b = &s->batch;
    b->id = ++s->serial;
    b->stage = s->pass;
    b->head = s->head;
    b->generation = s->generation;
    b->candidates = s->count;
    b->terms = s->params.terms;
    b->cols = s->cols;
    b->row_start = s->row;
    b->rows = s->rows - s->row < s->tile_rows ? s->rows - s->row : s->tile_rows;
    b->X = s->X + (size_t)s->row * s->cols;
    b->descriptors = s->descriptors;
    s->pending = 1;
    *out = b;
    return 1;
}
const sym_family_batch_t *sym_family_search_pending(const sym_family_search_t *s) {
    return s && s->pending ? &s->batch : NULL;
}
int sym_family_search_score(const sym_family_search_t *s, uint32_t id, double *out,
                            int32_t count) {
    if (!s || !s->pending || id != s->batch.id || !out ||
        count != s->count * s->batch.rows * s->params.terms)
        return fail("invalid family score batch");
    for (int c = 0; c < s->count; c++) {
        for (int r = 0; r < s->batch.rows; r++) {
            double *dst = out + ((size_t)c * s->batch.rows + r) * s->params.terms;
            if (features(&s->working[c], s->batch.X + (size_t)r * s->cols, s->params.terms,
                         dst))
                return fail("nonfinite family feature");
        }
    }
    return 0;
}
int sym_family_search_accept(sym_family_search_t *s, uint32_t id, const double *data,
                             int32_t count) {
    if (!s || !s->pending || id != s->batch.id || !data ||
        count != s->count * s->batch.rows * s->params.terms)
        return fail("invalid family accept batch");
    // Validate the entire tile before changing solver state; invalid acceptance
    // is retryable with the same identity and cannot partially advance QR.
    for (int i = 0; i < count; i++)
        if (!isfinite(data[i]))
            return fail("nonfinite family feature tile");
    int terms = s->params.terms;
    for (int c = 0; c < s->count; c++) {
        family_qr *q = &s->qr[c];
        candidate *cand = &s->working[c];
        for (int r = 0; r < s->batch.rows; r++) {
            int row = s->row + r, validation = s->validation[row];
            const double *phi = data + ((size_t)c * s->batch.rows + r) * terms;
            double target = family_target(s, row);
            if (s->pass == 0 && !validation) {
                q->n++;
                q->ym += (target - q->ym) / q->n;
                for (int j = 0; j < terms; j++) {
                    double d = phi[j] - q->mean[j];
                    q->mean[j] += d / q->n;
                    q->moment[j] += d * (phi[j] - q->mean[j]);
                }
            } else if (s->pass == 1 && !validation)
                qr_row(q, phi, target, terms);
            else if (s->pass == 2) {
                double pred = prediction(cand, phi, terms), delta = pred - target;
                if (!isfinite(f32(pred)) || !isfinite(delta * delta))
                    q->invalid = 1;
                if (validation) {
                    q->valid += delta * delta;
                    q->nv++;
                } else
                    q->train += delta * delta;
            }
        }
    }
    s->pending = 0;
    s->row += s->batch.rows;
    if (s->row != s->rows)
        return 0;
    s->row = 0;
    for (int c = 0; c < s->count; c++) {
        family_qr *q = &s->qr[c];
        candidate *cand = &s->working[c];
        if (s->pass == 0) {
            for (int j = 0; j < terms; j++) {
                q->scale[j] = sqrt(fmax(0, q->moment[j] / q->n));
                if (q->scale[j] == 0)
                    q->scale[j] = 1;
                q->R[j * terms + j] = sqrt(s->params.ridge * q->n) / q->scale[j];
            }
        } else if (s->pass == 1)
            qr_solve(q, cand, terms);
        else {
            cand->complexity = sym_family_complexity(cand, terms);
            if (!q->invalid) {
                cand->train = q->train / q->n;
                cand->valid = q->nv ? q->valid / q->nv : cand->train;
                cand->objective = cand->valid + s->params.complexity_penalty * cand->complexity;
                if (!isfinite(cand->objective))
                    cand->objective = DBL_MAX;
            }
            cand->ready = 1;
            s->evaluations++;
            if (s->phase == FAMILY_POPULATION) {
                s->population[s->slots[c]] = *cand;
                archive_candidate(s->model, s->head, cand);
            } else if (cand->objective + s->params.tol < s->selected.objective) {
                s->selected = *cand;
                s->polish_improved = 1;
            }
        }
    }
    if (++s->pass == 3) {
        s->count = 0;
        if (s->phase == FAMILY_POLISH) {
            if (s->polish_sign < 0)
                s->polish_sign = 1;
            else {
                s->polish_sign = -1;
                if (++s->polish_param == 2) {
                    s->polish_param = 0;
                    s->polish_term++;
                }
            }
        }
    }
    return 0;
}
uint64_t sym_family_search_evaluations(const sym_family_search_t *s) {
    return s ? s->evaluations : 0;
}
sym_family_model_t *sym_family_search_finish(sym_family_search_t *s) {
    if (!s || s->phase != FAMILY_DONE || s->pending) {
        fail("family search is not finished");
        return NULL;
    }
    sym_family_model_t *m = s->model;
    s->model = NULL;
    s->phase = FAMILY_TAKEN;
    return m;
}
sym_family_model_t *sym_family_fit(const double *X, int32_t rows, int32_t cols, const double *y,
                                   int32_t task, int32_t classes,
                                   const sym_family_params_t *p) {
    enum { FIT_CANDIDATES = 8, FIT_TILE_ROWS = 128 };
    sym_family_search_t *s = sym_family_search_new(
        X, rows, cols, y, task, classes, p, FIT_CANDIDATES, FIT_TILE_ROWS);
    if (!s)
        return NULL;
    /* Size all storage from the accepted search shape, never a second default. */
    const size_t row_values = (size_t)s->capacity * s->params.terms;
    const size_t tile_rows = (size_t)s->tile_rows;
    double *features = malloc(row_values * tile_rows * sizeof(double));
    if (!features) {
        sym_family_search_free(s);
        fail("out of memory allocating family tile");
        return NULL;
    }
    /* Cache a prefix of whole row tiles for the current candidate chunk.
     * The bound excludes the separately sized tile scratch buffer. Allocation
     * failure only loses this optimization; it must not make a fit fail. */
    const size_t cache_limit = 8u * 1024u * 1024u;
    size_t cache_rows = cache_limit / (row_values * sizeof(double));
    cache_rows = cache_rows >= (size_t)rows ? (size_t)rows : cache_rows / tile_rows * tile_rows;
    double *cache = malloc(cache_rows * row_values * sizeof(double));
    if (!cache)
        cache_rows = 0;
    const sym_family_batch_t *batch;
    int rc;
    while ((rc = sym_family_search_propose(s, &batch)) > 0) {
        int count = batch->candidates * batch->rows * batch->terms;
        int cached = (size_t)batch->row_start + batch->rows <= cache_rows;
        /* Tile-major storage preserves the ABI's candidate/row/term layout,
         * including partial candidate chunks. Stage zero overwrites all cached
         * tiles before QR/loss replay; each polish trial starts a new stage zero. */
        double *tile = cached ? cache + (size_t)batch->row_start * batch->candidates * batch->terms
                              : features;
        if (((!cached || batch->stage == 0) &&
             sym_family_search_score(s, batch->id, tile, count)) ||
            sym_family_search_accept(s, batch->id, tile, count)) {
            rc = -1;
            break;
        }
    }
    free(cache);
    sym_family_model_t *m = rc == 0 ? sym_family_search_finish(s) : NULL;
    free(features);
    sym_family_search_free(s);
    return m;
}

int sym_family_archive_size(const sym_family_model_t *m, int32_t h) {
    return m && h >= 0 && h < m->heads ? m->counts[h] : -1;
}
int sym_family_export(const sym_family_model_t *m, int32_t h, int32_t index, double *data,
                      int32_t count) {
    if (!m || h < 0 || h >= m->heads || !m->fitted[h] || index < -1 || index >= m->counts[h] ||
        !data || count < 6 * m->terms + 6)
        return fail("invalid family export");
    const candidate *c = index < 0 ? &m->best[h] : &m->archive[(size_t)h * m->frontier + index];
    for (int t = 0; t < m->terms; t++) {
        data[6 * t] = c->a[t];
        data[6 * t + 1] = c->b[t];
        data[6 * t + 2] = c->op[t];
        data[6 * t + 3] = c->p0[t];
        data[6 * t + 4] = c->p1[t];
        data[6 * t + 5] = c->coef[t];
    }
    double *tail = data + 6 * m->terms;
    tail[0] = c->bias;
    tail[1] = c->train;
    tail[2] = c->valid;
    tail[3] = c->complexity;
    tail[4] = c->objective;
    tail[5] = c->generation;
    return 6 * m->terms + 6;
}
int sym_family_import(sym_family_model_t *m, int32_t h, int32_t index, const double *data,
                      int32_t count) {
    if (!m || h < 0 || h >= m->heads || index < -1 || index >= m->frontier ||
        index > m->counts[h] || !data || count != 6 * m->terms + 6)
        return fail("invalid family import shape");
    candidate c = {0};
    for (int i = 0; i < count; i++)
        if (!isfinite(data[i]))
            return fail("nonfinite family artifact value");
    for (int t = 0; t < m->terms; t++) {
        const double *a = data + 6 * t;
        if (a[0] < 0 || a[0] >= m->cols || floor(a[0]) != a[0] || a[1] < 0 || a[1] >= m->cols ||
            floor(a[1]) != a[1] || a[2] < 0 || a[2] > 9 || floor(a[2]) != a[2] ||
            !isfinite(f32(a[3])) || !isfinite(f32(a[4])) || !isfinite(f32(a[5])))
            return fail("invalid family term");
        c.a[t] = (int32_t)a[0];
        c.b[t] = (int32_t)a[1];
        c.op[t] = (int32_t)a[2];
        c.p0[t] = f32(a[3]);
        c.p1[t] = f32(a[4]);
        c.coef[t] = f32(a[5]);
    }
    const double *a = data + 6 * m->terms;
    if (a[1] < 0 || a[2] < 0 || a[3] < 0 || a[4] < 0 || a[5] < 0 || a[5] > INT32_MAX ||
        floor(a[5]) != a[5])
        return fail("invalid family metrics");
    c.bias = a[0];
    c.train = a[1];
    c.valid = a[2];
    c.complexity = a[3];
    c.objective = a[4];
    c.generation = (int32_t)a[5];
    if (m->semantics == 2) {
        for (int t = 0; t < m->terms; t++) {
            const double *v = data + 6 * t;
            if (v[3] != c.p0[t] || v[4] != c.p1[t] || v[5] != c.coef[t] ||
                (c.op[t] < 4 && (c.p0[t] != 1 || c.p1[t] != 0)) ||
                (c.op[t] >= 4 && c.b[t] != 0))
                return fail("noncanonical family artifact term");
        }
        if (c.complexity != sym_family_complexity(&c, m->terms) ||
            c.objective != c.valid + m->penalty * c.complexity)
            return fail("inconsistent family artifact metrics");
    }
    canonicalize(&c, m->terms);
    if (index < 0) {
        m->best[h] = c;
        m->fitted[h] = 1;
    } else {
        m->archive[(size_t)h * m->frontier + index] = c;
        if (index == m->counts[h])
            m->counts[h]++;
    }
    return 0;
}
int sym_family_predict(const sym_family_model_t *m, const double *X, int32_t rows, int32_t cols,
                       int32_t mode, double *out) {
    if (!m || !X || !out || rows < 0 || cols != m->cols || mode < 0 || mode > 2 ||
        (!m->classes && mode == 2) || (size_t)rows > SIZE_MAX / sizeof(double) / (size_t)cols ||
        (size_t)rows > SIZE_MAX / sizeof(double) / (size_t)(m->classes ? m->classes : 1))
        return fail("invalid family prediction shape or mode");
    for (int h = 0; h < m->heads; h++)
        if (!m->fitted[h])
            return fail("family model is not fitted");
    for (size_t i = 0; i < (size_t)rows * cols; i++)
        if (!isfinite(f32(X[i])))
            return fail("family inputs must be finite float32 values");
    for (int i = 0; i < rows; i++) {
        double margins[128] = {0}, phi[SYM_FAMILY_MAX_TERMS];
        for (int h = 0; h < m->heads; h++) {
            if (features(&m->best[h], X + (size_t)i * cols, m->terms, phi))
                return fail("nonfinite family feature");
            margins[h] = f32(prediction(&m->best[h], phi, m->terms));
            if (!isfinite(margins[h]))
                return fail("family prediction overflow; rescale inputs");
        }
        if (!m->classes || mode == 1) {
            for (int h = 0; h < m->heads; h++)
                out[(size_t)i * m->heads + h] = margins[h];
        } else if (mode == 0) {
            int best = 0;
            for (int h = 1; h < m->heads; h++)
                if (margins[h] > margins[best])
                    best = h;
            out[i] = m->classes == 2 ? margins[0] >= 0 : best;
        } else if (m->classes == 2) {
            double prob = 1 / (1 + exp(-clamp(margins[0], -60, 60)));
            out[(size_t)2 * i] = 1 - prob;
            out[(size_t)2 * i + 1] = prob;
        } else {
            double maximum = margins[0], sum = 0;
            for (int h = 1; h < m->heads; h++)
                if (margins[h] > maximum)
                    maximum = margins[h];
            for (int h = 0; h < m->heads; h++) {
                margins[h] = exp(clamp(margins[h] - maximum, -60, 60));
                sum += margins[h];
            }
            for (int h = 0; h < m->heads; h++)
                out[(size_t)i * m->heads + h] = margins[h] / sum;
        }
    }
    return 0;
}

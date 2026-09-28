/* Benchmark-only translation unit: use the exact private product evaluator and
 * QR, rather than maintaining a second numerical reference. Never packaged. */
#include "../../src/sym_family.c"

int scorer_reference(const double *X, const double *y, const uint8_t *validation,
                     const double *descriptors, int rows, int cols, int batch, int terms,
                     double ridge, double *coefficients, double *losses, double *all_features) {
    double *phi = malloc((size_t)rows * terms * sizeof(double));
    if (!phi)
        return -1;
    for (int b = 0; b < batch; b++) {
        family_qr q = {0};
        candidate c = {0};
        for (int j = 0; j < terms; j++) {
            const double *d = descriptors + ((size_t)b * terms + j) * 5;
            c.a[j] = (int)d[0];
            c.b[j] = (int)d[1];
            c.op[j] = (int)d[2];
            c.p0[j] = f32(d[3]);
            c.p1[j] = f32(d[4]);
        }
        for (int r = 0; r < rows; r++) {
            if (features(&c, X + (size_t)r * cols, cols, terms, phi + (size_t)r * terms)) {
                free(phi);
                return -2;
            }
            if (validation[r])
                continue;
            q.n++;
            q.ym += (y[r] - q.ym) / q.n;
            for (int j = 0; j < terms; j++) {
                double v = phi[(size_t)r * terms + j], delta = v - q.mean[j];
                q.mean[j] += delta / q.n;
                q.moment[j] += delta * (v - q.mean[j]);
            }
        }
        if (!q.n) {
            free(phi);
            return -3;
        }
        for (int j = 0; j < terms; j++) {
            q.scale[j] = sqrt(fmax(0, q.moment[j] / q.n));
            if (!q.scale[j])
                q.scale[j] = 1;
            q.R[j * terms + j] = sqrt(ridge * q.n) / q.scale[j];
        }
        for (int r = 0; r < rows; r++)
            if (!validation[r])
                qr_row(&q, phi + (size_t)r * terms, y[r], terms);
        qr_solve(&q, &c, terms);
        for (int r = 0; r < rows; r++) {
            double delta = prediction(&c, phi + (size_t)r * terms, terms) - y[r];
            if (validation[r]) {
                q.valid += delta * delta;
                q.nv++;
            } else
                q.train += delta * delta;
        }
        memcpy(coefficients + (size_t)b * (terms + 1), c.coef, terms * sizeof(double));
        coefficients[(size_t)b * (terms + 1) + terms] = c.bias;
        losses[2 * b] = q.train / q.n;
        losses[2 * b + 1] = q.nv ? q.valid / q.nv : losses[2 * b];
        if (all_features)
            memcpy(all_features + (size_t)b * rows * terms, phi,
                   (size_t)rows * terms * sizeof(double));
    }
    free(phi);
    return 0;
}

/* stats stores anchor, shifted mean, scale. Device QR is of centered/scaled
 * [features,target], with validation rows zeroed. Apply original-unit ridge ONCE
 * to the small R, then use the same C solve and coefficient rounding as fit. */
int scorer_solve(const double *factors, const double *stats, int batch, int terms, int width,
                 int train_rows, double ym, double ridge, double *out) {
    for (int b = 0; b < batch; b++) {
        family_qr q = {0};
        candidate c = {0};
        const double *st = stats + (size_t)b * terms * 3;
        const double *R = factors + (size_t)b * width * width;
        for (int j = 0; j < terms; j++) {
            if (!(st[2 * terms + j] > 0))
                return -1;
            q.scale[j] = 1;
            q.R[j * terms + j] = sqrt(train_rows * ridge) / st[2 * terms + j];
        }
        for (int r = 0; r < width; r++)
            qr_row(&q, R + (size_t)r * width, R[(size_t)r * width + terms], terms);
        q.ym = ym;
        for (int j = 0; j < terms; j++) {
            q.mean[j] = st[j] + st[terms + j];
            q.scale[j] = st[2 * terms + j];
        }
        qr_solve(&q, &c, terms);
        if (q.invalid)
            return -2;
        memcpy(out + (size_t)b * (terms + 1), c.coef, terms * sizeof(double));
        out[(size_t)b * (terms + 1) + terms] = c.bias;
    }
    return 0;
}

static void scorer_params(sym_family_params_t *p, int population, int generations, int terms,
                          uint32_t seed, int polish) {
    sym_family_params_init(p);
    p->population = population;
    p->generations = generations;
    p->terms = terms;
    p->seed = seed;
    p->polish = polish;
    p->validation_fraction = .2;
}

sym_family_search_t *scorer_search_new(const double *X, const double *y, int rows, int cols,
                                       int population, int generations, int terms,
                                       uint32_t seed, int polish) {
    sym_family_params_t p;
    scorer_params(&p, population, generations, terms, seed, polish);
    return sym_family_search_new(X, rows, cols, y, 0, 0, &p, 64, rows < 4096 ? rows : 4096);
}

sym_family_model_t *scorer_fit(const double *X, const double *y, int rows, int cols,
                               int population, int generations, int terms, uint32_t seed,
                               int polish) {
    sym_family_params_t p;
    scorer_params(&p, population, generations, terms, seed, polish);
    return sym_family_fit(X, rows, cols, y, 0, 0, &p);
}

void scorer_search_mask(const sym_family_search_t *s, uint8_t *mask) {
    memcpy(mask, s->validation, s->rows);
}

int scorer_search_next(sym_family_search_t *s, double *descriptors, uint32_t *id) {
    const sym_family_batch_t *batch;
    int rc = sym_family_search_propose(s, &batch);
    if (rc <= 0)
        return rc;
    if (batch->stage || batch->row_start)
        return -1;
    memcpy(descriptors, batch->descriptors,
           (size_t)batch->candidates * batch->terms * 5 * sizeof(double));
    *id = batch->id;
    return batch->candidates;
}

/* Experimental bridge, compiled ONLY into the benchmark. No public ABI is added.
 * The search still owns all proposals, randomness, archive updates and polish.
 * Only a complete finite fitted result may replace the pending stage-zero work. */
int scorer_search_accept(sym_family_search_t *s, uint32_t id, const double *coef,
                         const double *loss) {
    if (!s || !s->pending || s->batch.id != id || s->pass || s->row)
        return -1;
    int terms = s->params.terms, nv = 0;
    for (int r = 0; r < s->rows; r++)
        nv += s->validation[r] != 0;
    for (int c = 0; c < s->count; c++) {
        for (int j = 0; j <= terms; j++)
            if (!isfinite(coef[(size_t)c * (terms + 1) + j]))
                return -2;
        if (!isfinite(loss[2 * c]) || !isfinite(loss[2 * c + 1]) || loss[2 * c] < 0 ||
            loss[2 * c + 1] < 0)
            return -2;
    }
    for (int c = 0; c < s->count; c++) {
        family_qr *q = &s->qr[c];
        memcpy(s->working[c].coef, coef + (size_t)c * (terms + 1), terms * sizeof(double));
        s->working[c].bias = coef[(size_t)c * (terms + 1) + terms];
        q->n = s->rows - nv;
        q->nv = nv;
        q->train = loss[2 * c] * q->n;
        q->valid = loss[2 * c + 1] * nv;
    }
    s->pending = 0;
    s->pass = 2;
    family_finish_pass(s);
    return 0;
}

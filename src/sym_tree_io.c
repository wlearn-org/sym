#include "sym.h"
#include "sym_internal.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} byte_writer_t;

static int bw_reserve(byte_writer_t *w, size_t add) {
    if (w->len + add <= w->cap) return 0;
    size_t nc = w->cap ? w->cap * 2 : 1024;
    while (nc < w->len + add) nc *= 2;
    uint8_t *p = (uint8_t *)realloc(w->data, nc);
    if (!p) {
        sym_set_error("out of memory serializing model");
        return -1;
    }
    w->data = p;
    w->cap = nc;
    return 0;
}

static int bw_write(byte_writer_t *w, const void *src, size_t n) {
    if (bw_reserve(w, n) != 0) return -1;
    memcpy(w->data + w->len, src, n);
    w->len += n;
    return 0;
}

static int bw_i32(byte_writer_t *w, int32_t v) { return bw_write(w, &v, sizeof(v)); }
static int bw_u32(byte_writer_t *w, uint32_t v) { return bw_write(w, &v, sizeof(v)); }
static int bw_f64(byte_writer_t *w, double v) { return bw_write(w, &v, sizeof(v)); }

static int write_params(byte_writer_t *w, const sym_params_t *p) {
    return bw_i32(w, p->task) || bw_i32(w, p->population) || bw_i32(w, p->generations) ||
        bw_i32(w, p->max_nodes) || bw_i32(w, p->max_depth) || bw_i32(w, p->frontier_size) ||
        bw_i32(w, p->tournament_size) || bw_i32(w, p->elite_count) || bw_i32(w, p->top_k) ||
        bw_i32(w, p->n_classes) || bw_i32(w, p->loss) || bw_i32(w, p->operator_set) ||
        bw_i32(w, p->early_stop_rounds) || bw_u32(w, p->seed) ||
        bw_f64(w, p->validation_fraction) || bw_f64(w, p->complexity_penalty) ||
        bw_f64(w, p->mutation_rate) || bw_f64(w, p->crossover_rate) ||
        bw_f64(w, p->constant_rate) || bw_f64(w, p->const_min) || bw_f64(w, p->const_max) ||
        bw_f64(w, p->huber_delta) || bw_f64(w, p->tol) ||
        bw_i32(w, p->islands) || bw_i32(w, p->migration_interval) ||
        bw_i32(w, p->migration_count) || bw_i32(w, p->warmup_generations) ||
        bw_i32(w, p->warmup_min_nodes) || bw_i32(w, p->brood_size) ||
        bw_i32(w, p->row_sample_size) || bw_i32(w, p->local_refine_interval) ||
        bw_i32(w, p->local_refine_count) || bw_i32(w, p->complexity_hof_size) ||
        bw_i32(w, p->final_selector) || bw_f64(w, p->complexity_bucket_width);
}

static int write_formula(byte_writer_t *w, const sym_formula_t *f) {
    if (bw_i32(w, f->n_nodes) || bw_i32(w, f->capacity) ||
        bw_f64(w, f->train_loss) || bw_f64(w, f->valid_loss) ||
        bw_f64(w, f->objective) || bw_f64(w, f->complexity)) return -1;
    for (int32_t i = 0; i < f->n_nodes; i++) {
        const sym_node_t *n = &f->nodes[i];
        if (bw_i32(w, n->op) || bw_i32(w, n->left) || bw_i32(w, n->right) ||
            bw_i32(w, n->feature) || bw_f64(w, n->value)) return -1;
    }
    return 0;
}

int sym_save(const sym_model_t *model, char **out_buf, int32_t *out_len) {
    if (!model || !out_buf || !out_len) {
        sym_set_error("invalid save arguments");
        return -1;
    }
    byte_writer_t w;
    memset(&w, 0, sizeof(w));
    if (bw_write(&w, SYM_MAGIC, 4) || bw_u32(&w, SYM_VERSION) ||
        bw_i32(&w, model->task) || bw_i32(&w, model->n_features) ||
        bw_i32(&w, model->n_classes) || bw_i32(&w, model->n_outputs) ||
        bw_u32(&w, model->seed) || write_params(&w, &model->params) ||
        bw_i32(&w, model->n_formulas) || bw_i32(&w, model->n_frontier)) {
        free(w.data);
        return -1;
    }
    for (int32_t i = 0; i < model->n_formulas; i++) {
        if (write_formula(&w, &model->formulas[i]) != 0) {
            free(w.data);
            return -1;
        }
    }
    for (int32_t i = 0; i < model->n_frontier; i++) {
        if (write_formula(&w, &model->frontier[i]) != 0) {
            free(w.data);
            return -1;
        }
    }
    *out_buf = (char *)w.data;
    *out_len = (int32_t)w.len;
    return 0;
}

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t pos;
} byte_reader_t;

static int br_read(byte_reader_t *r, void *dst, size_t n) {
    if (r->pos + n > r->len) {
        sym_set_error("truncated symbolic model artifact");
        return -1;
    }
    memcpy(dst, r->data + r->pos, n);
    r->pos += n;
    return 0;
}

static int br_i32(byte_reader_t *r, int32_t *v) { return br_read(r, v, sizeof(*v)); }
static int br_u32(byte_reader_t *r, uint32_t *v) { return br_read(r, v, sizeof(*v)); }
static int br_f64(byte_reader_t *r, double *v) { return br_read(r, v, sizeof(*v)); }

static int read_params(byte_reader_t *r, sym_params_t *p) {
    return br_i32(r, &p->task) || br_i32(r, &p->population) || br_i32(r, &p->generations) ||
        br_i32(r, &p->max_nodes) || br_i32(r, &p->max_depth) || br_i32(r, &p->frontier_size) ||
        br_i32(r, &p->tournament_size) || br_i32(r, &p->elite_count) || br_i32(r, &p->top_k) ||
        br_i32(r, &p->n_classes) || br_i32(r, &p->loss) || br_i32(r, &p->operator_set) ||
        br_i32(r, &p->early_stop_rounds) || br_u32(r, &p->seed) ||
        br_f64(r, &p->validation_fraction) || br_f64(r, &p->complexity_penalty) ||
        br_f64(r, &p->mutation_rate) || br_f64(r, &p->crossover_rate) ||
        br_f64(r, &p->constant_rate) || br_f64(r, &p->const_min) || br_f64(r, &p->const_max) ||
        br_f64(r, &p->huber_delta) || br_f64(r, &p->tol) ||
        br_i32(r, &p->islands) || br_i32(r, &p->migration_interval) ||
        br_i32(r, &p->migration_count) || br_i32(r, &p->warmup_generations) ||
        br_i32(r, &p->warmup_min_nodes) || br_i32(r, &p->brood_size) ||
        br_i32(r, &p->row_sample_size) || br_i32(r, &p->local_refine_interval) ||
        br_i32(r, &p->local_refine_count) || br_i32(r, &p->complexity_hof_size) ||
        br_i32(r, &p->final_selector) || br_f64(r, &p->complexity_bucket_width);
}

static int read_formula(byte_reader_t *r, sym_formula_t *f, int32_t n_features) {
    int32_t n_nodes = 0;
    int32_t capacity = 0;
    if (br_i32(r, &n_nodes) || br_i32(r, &capacity)) return -1;
    if (n_nodes <= 0 || n_nodes > 255 || capacity < n_nodes || capacity > 255) {
        sym_set_error("invalid formula size in symbolic artifact");
        return -1;
    }
    if (sym_tree_formula_init(f, capacity) != 0) return -1;
    f->n_nodes = n_nodes;
    if (br_f64(r, &f->train_loss) || br_f64(r, &f->valid_loss) ||
        br_f64(r, &f->objective) || br_f64(r, &f->complexity)) {
        sym_tree_formula_free(f);
        return -1;
    }
    for (int32_t i = 0; i < n_nodes; i++) {
        sym_node_t *n = &f->nodes[i];
        if (br_i32(r, &n->op) || br_i32(r, &n->left) || br_i32(r, &n->right) ||
            br_i32(r, &n->feature) || br_f64(r, &n->value)) {
            sym_tree_formula_free(f);
            return -1;
        }
    }
    if (sym_tree_formula_validate(f, n_features) != 0) {
        sym_tree_formula_free(f);
        return -1;
    }
    return 0;
}

sym_model_t *sym_load(const char *buf, int32_t len) {
    if (!buf || len < 8) {
        sym_set_error("invalid load buffer");
        return NULL;
    }
    byte_reader_t r;
    r.data = (const uint8_t *)buf;
    r.len = (size_t)len;
    r.pos = 0;
    char magic[4];
    uint32_t version = 0;
    if (br_read(&r, magic, 4) || memcmp(magic, SYM_MAGIC, 4) != 0 || br_u32(&r, &version) || version != SYM_VERSION) {
        sym_set_error("unsupported symbolic model artifact");
        return NULL;
    }
    sym_model_t *m = (sym_model_t *)calloc(1, sizeof(sym_model_t));
    if (!m) {
        sym_set_error("out of memory loading model");
        return NULL;
    }
    if (br_i32(&r, &m->task) || br_i32(&r, &m->n_features) || br_i32(&r, &m->n_classes) ||
        br_i32(&r, &m->n_outputs) || br_u32(&r, &m->seed) || read_params(&r, &m->params) ||
        br_i32(&r, &m->n_formulas) || br_i32(&r, &m->n_frontier)) {
        sym_free(m);
        return NULL;
    }
    if (m->n_features <= 0 || m->n_outputs <= 0 || m->n_outputs > 128 ||
        m->n_formulas != m->n_outputs || m->n_frontier < 0 || m->n_frontier > 128) {
        sym_set_error("invalid symbolic model header");
        sym_free(m);
        return NULL;
    }
    m->formulas = (sym_formula_t *)calloc((size_t)m->n_formulas, sizeof(sym_formula_t));
    m->frontier = (sym_formula_t *)calloc((size_t)(m->n_frontier > 0 ? m->n_frontier : 1), sizeof(sym_formula_t));
    if (!m->formulas || !m->frontier) {
        sym_set_error("out of memory loading formulas");
        sym_free(m);
        return NULL;
    }
    for (int32_t i = 0; i < m->n_formulas; i++) {
        if (read_formula(&r, &m->formulas[i], m->n_features) != 0) {
            sym_free(m);
            return NULL;
        }
    }
    for (int32_t i = 0; i < m->n_frontier; i++) {
        if (read_formula(&r, &m->frontier[i], m->n_features) != 0) {
            sym_free(m);
            return NULL;
        }
    }
    return m;
}

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} str_writer_t;

static int sw_reserve(str_writer_t *w, size_t add) {
    if (!w) return -1;
    if (add > SIZE_MAX - w->len - 1) {
        sym_set_error("formula text too large");
        return -1;
    }
    size_t need = w->len + add + 1;
    if (w->data && need <= w->cap) return 0;
    size_t nc = w->cap ? w->cap * 2 : 512;
    while (nc < need) {
        if (nc > SIZE_MAX / 2) {
            sym_set_error("formula text too large");
            return -1;
        }
        nc *= 2;
    }
    char *p = (char *)realloc(w->data, nc);
    if (!p) {
        sym_set_error("out of memory formatting formula");
        return -1;
    }
    w->data = p;
    w->cap = nc;
    return 0;
}

static int sw_append(str_writer_t *w, const char *s) {
    size_t n = strlen(s);
    if (sw_reserve(w, n) != 0) return -1;
    memcpy(w->data + w->len, s, n);
    w->len += n;
    w->data[w->len] = '\0';
    return 0;
}

static int sw_printf(str_writer_t *w, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[128];
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return -1;
    if ((size_t)n < sizeof(tmp)) return sw_append(w, tmp);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) return -1;
    va_start(ap, fmt);
    vsnprintf(buf, (size_t)n + 1, fmt, ap);
    va_end(ap);
    int rc = sw_append(w, buf);
    free(buf);
    return rc;
}

static int formula_text_rec(const sym_formula_t *f, int32_t idx, str_writer_t *w, int depth) {
    if (depth > SYM_MAX_EXPR_RECURSION || idx < 0 || idx >= f->n_nodes) return sw_append(w, "?");
    const sym_node_t *n = &f->nodes[idx];
    switch (n->op) {
        case SYM_OP_CONST:
            return sw_printf(w, "%.12g", n->value);
        case SYM_OP_VAR:
            return sw_printf(w, "x%d", n->feature);
        case SYM_OP_ADD:
        case SYM_OP_SUB:
        case SYM_OP_MUL:
        case SYM_OP_DIV:
        case SYM_OP_MIN:
        case SYM_OP_MAX: {
            const char *sym = n->op == SYM_OP_ADD ? " + " :
                              n->op == SYM_OP_SUB ? " - " :
                              n->op == SYM_OP_MUL ? " * " :
                              n->op == SYM_OP_DIV ? " / " :
                              n->op == SYM_OP_MIN ? ", " : ", ";
            if (n->op == SYM_OP_MIN || n->op == SYM_OP_MAX) {
                if (sw_append(w, n->op == SYM_OP_MIN ? "min(" : "max(")) return -1;
                if (formula_text_rec(f, n->left, w, depth + 1)) return -1;
                if (sw_append(w, sym)) return -1;
                if (formula_text_rec(f, n->right, w, depth + 1)) return -1;
                return sw_append(w, ")");
            }
            if (sw_append(w, "(")) return -1;
            if (formula_text_rec(f, n->left, w, depth + 1)) return -1;
            if (sw_append(w, sym)) return -1;
            if (formula_text_rec(f, n->right, w, depth + 1)) return -1;
            return sw_append(w, ")");
        }
        case SYM_OP_NEG:
            if (sw_append(w, "-(")) return -1;
            if (formula_text_rec(f, n->left, w, depth + 1)) return -1;
            return sw_append(w, ")");
        default:
            if (sw_printf(w, "%s(", sym_tree_op_name(n->op))) return -1;
            if (formula_text_rec(f, n->left, w, depth + 1)) return -1;
            return sw_append(w, ")");
    }
}

static const sym_formula_t *select_formula(const sym_model_t *model, int32_t index) {
    if (!model || index < 0) return NULL;
    if (index < model->n_formulas) return &model->formulas[index];
    int32_t fi = index - model->n_formulas;
    if (fi >= 0 && fi < model->n_frontier) return &model->frontier[fi];
    return NULL;
}

static sym_formula_t *select_formula_mut(sym_model_t *model, int32_t index) {
    if (!model || index < 0) return NULL;
    if (index < model->n_formulas) return &model->formulas[index];
    int32_t fi = index - model->n_formulas;
    if (fi >= 0 && fi < model->n_frontier) return &model->frontier[fi];
    return NULL;
}

int sym_formula_text(const sym_model_t *model, int32_t formula_index, char **out_buf, int32_t *out_len) {
    if (!out_buf || !out_len) return -1;
    const sym_formula_t *f = select_formula(model, formula_index);
    if (!f) {
        sym_set_error("formula index out of range");
        return -1;
    }
    str_writer_t w;
    memset(&w, 0, sizeof(w));
    if (formula_text_rec(f, f->n_nodes - 1, &w, 0) != 0) {
        free(w.data);
        return -1;
    }
    *out_buf = w.data;
    *out_len = (int32_t)w.len;
    return 0;
}

int sym_formula_json(const sym_model_t *model, int32_t formula_index, char **out_buf, int32_t *out_len) {
    if (!out_buf || !out_len) return -1;
    const sym_formula_t *f = select_formula(model, formula_index);
    if (!f) {
        sym_set_error("formula index out of range");
        return -1;
    }
    str_writer_t w;
    memset(&w, 0, sizeof(w));
    if (sw_printf(&w, "{\"complexity\":%.17g,\"nodes\":[", f->complexity)) goto fail;
    for (int32_t i = 0; i < f->n_nodes; i++) {
        const sym_node_t *n = &f->nodes[i];
        if (i && sw_append(&w, ",")) goto fail;
        if (sw_printf(&w,
            "{\"feature\":%d,\"left\":%d,\"op\":\"%s\",\"opId\":%d,\"right\":%d,\"value\":%.17g}",
            n->feature, n->left, sym_tree_op_name(n->op), n->op, n->right, n->value)) goto fail;
    }
    if (sw_printf(&w, "],\"objective\":%.17g,\"trainLoss\":%.17g,\"validLoss\":%.17g}",
                  f->objective, f->train_loss, f->valid_loss)) goto fail;
    *out_buf = w.data;
    *out_len = (int32_t)w.len;
    return 0;
fail:
    free(w.data);
    return -1;
}

int sym_set_formula_constant(
    sym_model_t *model,
    int32_t formula_index,
    int32_t node_index,
    double value
) {
    sym_formula_t *f = select_formula_mut(model, formula_index);
    if (!f) {
        sym_set_error("formula index out of range");
        return -1;
    }
    if (node_index < 0 || node_index >= f->n_nodes) {
        sym_set_error("formula node index out of range");
        return -1;
    }
    if (f->nodes[node_index].op != SYM_OP_CONST) {
        sym_set_error("formula node is not a constant");
        return -1;
    }
    if (!sym_tree_is_finite_value(value)) {
        sym_set_error("constant value must be finite");
        return -1;
    }
    f->nodes[node_index].value = value;
    return 0;
}

int sym_set_formula_metrics(
    sym_model_t *model,
    int32_t formula_index,
    double train_loss,
    double valid_loss,
    double objective
) {
    sym_formula_t *f = select_formula_mut(model, formula_index);
    if (!f) {
        sym_set_error("formula index out of range");
        return -1;
    }
    if (!sym_tree_is_finite_value(train_loss) || !sym_tree_is_finite_value(valid_loss) || !sym_tree_is_finite_value(objective)) {
        sym_set_error("formula metrics must be finite");
        return -1;
    }
    f->train_loss = train_loss;
    f->valid_loss = valid_loss;
    f->objective = objective;
    return 0;
}

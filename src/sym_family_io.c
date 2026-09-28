#include "sym_family_internal.h"
#include "sym_internal.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Fixed prefix: magic, version, kind, semantics, cols, classes, terms, frontier,
 * heads, penalty. Each head then stores its archive count, selected candidate,
 * and archive candidates in the public packed-candidate layout. */
#define PREFIX 44u
static uint32_t read_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static void write_u32(unsigned char *p, uint32_t v) {
    for (int i = 0; i < 4; i++)
        p[i] = (unsigned char)(v >> (8 * i));
}
static double read_f64(const unsigned char *p) {
    uint64_t bits = 0;
    for (int i = 0; i < 8; i++)
        bits |= (uint64_t)p[i] << (8 * i);
    double value;
    memcpy(&value, &bits, 8);
    return value;
}
static void write_f64(unsigned char *p, double value) {
    uint64_t bits;
    memcpy(&bits, &value, 8);
    for (int i = 0; i < 8; i++)
        p[i] = (unsigned char)(bits >> (8 * i));
}
int sym_family_dimensions(const sym_family_model_t *m, int32_t out[4]) {
    if (!m || !out) {
        sym_set_error("invalid family dimensions request");
        return -1;
    }
    out[0] = m->cols;
    out[1] = m->classes;
    out[2] = m->terms;
    out[3] = m->frontier;
    return 0;
}
int sym_family_save(const sym_family_model_t *m, char **out, int32_t *length) {
    if (!m || !out || !length || (m->semantics != 2 && m->semantics != 3)) {
        sym_set_error("SYM2 requires a fitted family model");
        return -1;
    }
    *out = NULL;
    *length = 0;
    int n = 6 * m->terms + 6;
    size_t total = PREFIX;
    for (int h = 0; h < m->heads; h++) {
        if (!m->fitted[h]) {
            sym_set_error("family model is not fitted");
            return -1;
        }
        total += 4 + (size_t)(m->counts[h] + 1) * n * 8;
    }
    if (total > INT32_MAX) {
        sym_set_error("family artifact exceeds size limit");
        return -1;
    }
    unsigned char *bytes = malloc(total);
    if (!bytes) {
        sym_set_error("out of memory saving family");
        return -1;
    }
    memcpy(bytes, "SYM2", 4);
    uint32_t fields[] = {2, 1, m->semantics, m->cols, m->classes, m->terms, m->frontier, m->heads};
    for (int i = 0; i < 8; i++)
        write_u32(bytes + 4 + 4 * i, fields[i]);
    write_f64(bytes + 36, m->penalty);
    size_t offset = PREFIX;
    double data[6 * SYM_FAMILY_MAX_TERMS + 6];
    for (int h = 0; h < m->heads; h++) {
        write_u32(bytes + offset, m->counts[h]);
        offset += 4;
        for (int at = -1; at < m->counts[h]; at++) {
            if (sym_family_export(m, h, at, data, n) != n) {
                free(bytes);
                return -1;
            }
            for (int j = 0; j < n; j++, offset += 8)
                write_f64(bytes + offset, data[j]);
        }
    }
    *out = (char *)bytes;
    *length = (int32_t)total;
    return 0;
}
sym_family_model_t *sym_family_load(const char *input, int32_t length) {
    if (!input || length < (int32_t)PREFIX) {
        sym_set_error("truncated family artifact");
        return NULL;
    }
    const unsigned char *bytes = (const unsigned char *)input;
    if (memcmp(bytes, "SYM2", 4) || read_u32(bytes + 4) != 2 || read_u32(bytes + 8) != 1 ||
        (read_u32(bytes + 12) != 2 && read_u32(bytes + 12) != 3)) {
        sym_set_error("unsupported family artifact version/kind/semantics");
        return NULL;
    }
    uint32_t cols = read_u32(bytes + 16), classes = read_u32(bytes + 20);
    uint32_t terms = read_u32(bytes + 24), frontier = read_u32(bytes + 28),
             heads = read_u32(bytes + 32);
    double penalty = read_f64(bytes + 36);
    if (!cols || cols > INT32_MAX || (read_u32(bytes + 12) == 3 && cols > INT32_MAX - terms) ||
        classes == 1 || classes > 128 || !terms || terms > SYM_FAMILY_MAX_TERMS || !frontier ||
        frontier > 128 || heads != (classes > 2 ? classes : 1) || !isfinite(penalty) ||
        penalty < 0) {
        sym_set_error("invalid family artifact header");
        return NULL;
    }
    int n = 6 * terms + 6;
    uint64_t offset = PREFIX;
    for (uint32_t h = 0; h < heads; h++) {
        if (offset + 4 > (uint64_t)length) {
            sym_set_error("truncated family head");
            return NULL;
        }
        uint32_t count = read_u32(bytes + (size_t)offset);
        if (count > frontier) {
            sym_set_error("invalid family archive count");
            return NULL;
        }
        offset += 4 + (uint64_t)(count + 1) * n * 8;
        if (offset > (uint64_t)length) {
            sym_set_error("truncated family candidates");
            return NULL;
        }
    }
    if (offset != (uint64_t)length) {
        sym_set_error("trailing family artifact data");
        return NULL;
    }
    sym_family_model_t *m = sym_family_new(cols, classes, terms, frontier);
    if (!m)
        return NULL;
    m->semantics = read_u32(bytes + 12);
    m->penalty = penalty;
    offset = PREFIX;
    double data[6 * SYM_FAMILY_MAX_TERMS + 6];
    for (uint32_t h = 0; h < heads; h++) {
        uint32_t count = read_u32(bytes + (size_t)offset);
        offset += 4;
        for (int at = -1; at < (int)count; at++) {
            for (int j = 0; j < n; j++, offset += 8)
                data[j] = read_f64(bytes + (size_t)offset);
            if (sym_family_import(m, h, at, data, n)) {
                sym_family_free(m);
                return NULL;
            }
        }
    }
    return m;
}

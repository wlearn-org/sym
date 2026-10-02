#ifndef WLEARN_SYM_FAMILY_INTERNAL_H
#define WLEARN_SYM_FAMILY_INTERNAL_H
#include "sym_family.h"

enum { SYM_FAMILY_FIT_CANDIDATES = 8, SYM_FAMILY_FIT_TILE_ROWS = 128 };

typedef struct {
    int32_t a[SYM_FAMILY_MAX_TERMS], b[SYM_FAMILY_MAX_TERMS], op[SYM_FAMILY_MAX_TERMS];
    double p0[SYM_FAMILY_MAX_TERMS], p1[SYM_FAMILY_MAX_TERMS], coef[SYM_FAMILY_MAX_TERMS];
    double bias, train, valid, complexity, objective;
    int32_t generation, ordinal;
    int ready;
} candidate;
struct sym_family_model {
    int32_t cols, classes, heads, terms, frontier, semantics;
    double penalty;
    candidate *best, *archive;
    int32_t *counts;
    uint8_t *fitted;
};

double sym_family_complexity(const candidate *c, int terms);
int sym_family_lm_step(const double *, int, int, const double *, double, double *);
#endif

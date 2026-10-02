#include "sym_family_internal.h"
#include <float.h>
#include <math.h>
#include <string.h>

/* Finite differences of the profiled residual (each probe refits its readout).
 * Columns include sqrt(n*ridge)*coefficients, so ridge participates in the step.
 * Column scaling and augmented Givens QR avoid forming J'J on collinear terms.
 * This is a bounded damped step, not MINPACK's trust-region implementation. */
int sym_family_lm_step(const double *residuals, int rows, int parameters, const double *increments,
                       double damping, double *step) {
    enum { MAX_P = 2 * SYM_FAMILY_MAX_TERMS };
    double R[MAX_P * MAX_P] = {0}, z[MAX_P] = {0}, scale[MAX_P] = {0};
    if (!residuals || !increments || !step || rows < 1 || parameters < 1 || parameters > MAX_P ||
        !isfinite(damping) || damping <= 0)
        return -1;
    for (int j = 0; j < parameters; j++) {
        if (!isfinite(increments[j]))
            return -1;
        for (int r = 0; r < rows; r++) {
            double value =
                increments[j] == 0
                    ? 0
                    : (residuals[(size_t)(j + 1) * rows + r] - residuals[r]) / increments[j];
            if (!isfinite(value) || !isfinite(residuals[r]))
                return -1;
            scale[j] = hypot(scale[j], value);
        }
        if (scale[j] == 0)
            scale[j] = 1;
        R[j * parameters + j] = sqrt(damping);
    }
    for (int r = 0; r < rows; r++) {
        double row[MAX_P], target = -residuals[r];
        for (int j = 0; j < parameters; j++)
            row[j] = increments[j] == 0 ? 0
                                        : (residuals[(size_t)(j + 1) * rows + r] - residuals[r]) /
                                              increments[j] / scale[j];
        for (int j = 0; j < parameters; j++) {
            double h = hypot(R[j * parameters + j], row[j]);
            double co = R[j * parameters + j] / h, si = row[j] / h;
            for (int k = j; k < parameters; k++) {
                double old = R[j * parameters + k];
                R[j * parameters + k] = co * old + si * row[k];
                row[k] = -si * old + co * row[k];
            }
            double old = z[j];
            z[j] = co * old + si * target;
            target = -si * old + co * target;
        }
    }
    for (int j = parameters - 1; j >= 0; j--) {
        double value = z[j];
        for (int k = j + 1; k < parameters; k++)
            value -= R[j * parameters + k] * step[k];
        step[j] = value / R[j * parameters + j];
    }
    double norm = 0;
    for (int j = 0; j < parameters; j++) {
        step[j] /= scale[j];
        if (!isfinite(step[j]))
            return -1;
        norm = hypot(norm, step[j]);
    }
    /* The bound is in the parameter coordinates supplied by the search owner. */
    if (norm > 2)
        for (int j = 0; j < parameters; j++)
            step[j] *= 2 / norm;
    return 0;
}

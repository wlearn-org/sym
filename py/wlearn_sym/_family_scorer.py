"""Resident family scoring through Polygrad's public frontend.

C owns proposals, masks, ridge, acceptance and portable prediction. The device
owns feature evaluation, stable centering, QR and explicit prediction losses.
"""

import time
import numpy as np
import polygrad as pg
from polygrad.uop.ops import AxisType, KernelInfo, UOp


def _lt(x, value):
    return x.lt(value) if isinstance(x, UOp) else x < value


def _abs(x):
    return _lt(x, 0).where(-x, x)


def _clamp(x, low, high):
    return _lt(x, low).where(low, _lt(x, high).where(x, high))


def _term(op, a, b, p0, p1, operators):
    """The current SYM2 family grammar, not symcpg's older compound grammar."""
    z = a * p0 + p1
    denominator = _lt(_abs(b), 1e-6).where(_lt(b, 0).where(-1e-6, 1e-6), b)
    branches = {
        0: lambda: a + b,
        1: lambda: a - b,
        2: lambda: a * b,
        3: lambda: a / denominator,
        4: lambda: z.sin(),
        5: lambda: (z + np.pi / 2).sin(),
        6: lambda: 2 / (1 + (-2 * _clamp(z, -20, 20) / np.log(2)).exp2()) - 1,
        7: lambda: (_abs(z) + 1e-6).log2() * np.log(2),
        8: lambda: _abs(z).sqrt(),
        9: lambda: (_clamp(z, -6, 6) / np.log(2)).exp2(),
    }
    value = branches[operators[0]]()
    for kind in operators[1:]:
        value = op.eq(kind).where(branches[kind](), value)
    return value


class FamilyScorer:
    """Own one runtime and reusable stages; arrays crossing back are O(batch*terms²)."""

    def __init__(
        self,
        X,
        y,
        validation,
        descriptors,
        *,
        solve,
        target_mean,
        runtime=None,
        hierarchical=False,
        mode="custom",
        device="auto",
        dtype="float64",
        ridge=1e-8,
        operators=tuple(range(10)),
    ):
        self.rt = None
        self.owned_runtime = runtime is None
        self.stages = []
        self.outputs = []
        self.calls = 0
        self.hierarchical = hierarchical
        self.solve = solve
        if dtype not in ("float32", "float64"):
            raise ValueError("scorerDtype must be float32 or float64")
        self.dtype = dtype
        self.npdtype = np.dtype(dtype)
        self.batch, self.terms, _ = descriptors.shape
        self.rows, self.cols = X.shape
        self.feature_bounds = np.max(np.abs(X.astype(np.float32).astype(float)), axis=0)
        self.width = 1 << self.terms.bit_length()
        self.ntrain = int(np.count_nonzero(validation == 0))
        self.nvalid = self.rows - self.ntrain
        if self.ntrain < 2:
            raise ValueError("family scoring requires at least two training rows")
        self.anchor_row = int(np.flatnonzero(validation == 0)[0])
        y = y.astype(np.float32).astype(float)
        self.ym = target_mean
        self.ridge = ridge
        self.operators = tuple(operators)
        self.device = None
        self.setup_start = time.perf_counter()
        self._check_cosine_phase(descriptors)
        try:
            self.rt = runtime or pg.create(device=device)
            device = self.device
            T = self.rt.Tensor
            self.X = T(X.astype(np.float32).astype(self.npdtype), device=device)
            self.y = T(y.astype(self.npdtype), device=device)
            self.mask = T((validation == 0).astype(self.npdtype), device=device)
            self.indices = T(descriptors[:, :, :3].astype(np.int32), device=device)
            self.params = T(
                descriptors[:, :, 3:].astype(np.float32).astype(self.npdtype),
                device=device,
            )
            self.coef = T(
                np.zeros((self.batch, self.terms + 1), dtype=self.npdtype),
                device=device,
            )
            feature_fn = (
                self._hierarchical_features
                if hierarchical
                else (
                    self._custom_features if mode == "custom" else self._tensor_features
                )
            )
            self.feature = self._compile(
                "features", feature_fn, [self.X, self.indices, self.params]
            )
            phi = self.feature.run([self.X, self.indices, self.params])
            self.normalize = self._compile(
                "normalize", self._normalize, [phi, self.y, self.mask]
            )
            design, stats = self.normalize.run([phi, self.y, self.mask])
            self.factor = self._compile("qr", lambda a: a.qr(mode="r"), [design])
            loss_fn = self._custom_loss if mode == "custom" else self._tensor_loss
            self.loss = self._compile(
                "loss", loss_fn, [phi, self.y, self.mask, self.coef]
            )
            self.setup_ms = (time.perf_counter() - self.setup_start) * 1000
        except BaseException:
            self.dispose()
            raise

    def _compile(self, name, function, inputs):
        def tracked(*args):
            result = function(*args)
            self.outputs.extend(
                result if isinstance(result, (list, tuple)) else [result]
            )
            return result

        stage = pg.compile(tracked, inputs)
        self.stages.append(stage)
        return stage

    def _empty(self, shape):
        return self.rt.Tensor.empty(shape, dtype=self.dtype, device=self.device)

    def _custom_features(self, X, indices, params):
        b, n, t, cols = self.batch, self.rows, self.terms, self.cols
        operators = self.operators

        def kernel(out, x, di, df):
            out, x, di, df = (v.flatten() for v in (out, x, di, df))
            i = UOp.range(out.ctx, b * n * t, 0)
            c, row, term = i // (n * t), (i // t) % n, i % t
            a = x[row * cols + di[(c * t + term) * 3]]
            v = x[row * cols + di[(c * t + term) * 3 + 1]]
            op = di[(c * t + term) * 3 + 2]
            p0, p1 = df[(c * t + term) * 2], df[(c * t + term) * 2 + 1]
            result = _term(op, a, v, p0, p1, operators)
            return (
                out[i]
                .store(result)
                .end(i)
                .sink(arg=KernelInfo(name="sym_family_terms"))
            )

        return self._empty((b, n, t)).custom_kernel(X, indices, params, fxn=kernel)[0]

    def _hierarchical_features(self, X, indices, params):
        b, n, t, cols = self.batch, self.rows, self.terms, self.cols

        def kernel(out, x, di, df):
            out, x, di, df = (v.flatten() for v in (out, x, di, df))
            i = UOp.range(out.ctx, b * n, 0)
            c, row = i // n, i % n
            values, stores = [], []
            for j in range(t):
                at = (c * t + j) * 3

                def source(index):
                    value = x[row * cols + index.lt(cols).where(index, 0)]
                    for earlier, prior in enumerate(values):
                        value = index.eq(cols + earlier).where(prior, value)
                    return value

                a, v = source(di[at]), source(di[at + 1])
                fp = (c * t + j) * 2
                value = _term(di[at + 2], a, v, df[fp], df[fp + 1], self.operators)
                values.append(value)
                stores.append(out[i * t + j].store(value))
            return (
                stores[0]
                .group(*stores[1:])
                .end(i)
                .sink(arg=KernelInfo(name="sym_family_hierarchy"))
            )

        return self._empty((b, n, t)).custom_kernel(X, indices, params, fxn=kernel)[0]

    def _tensor_features(self, X, indices, params):
        b, n, t = self.batch, self.rows, self.terms
        rows = self.rt.Tensor.arange(n, device=self.device).reshape(1, n, 1) * self.cols
        aindex = (indices[:, :, 0].reshape(b, 1, t) + rows).reshape(-1)
        bindex = (indices[:, :, 1].reshape(b, 1, t) + rows).reshape(-1)
        a = X.flatten().gather(0, aindex).reshape(b, n, t)
        v = X.flatten().gather(0, bindex).reshape(b, n, t)
        op = indices[:, :, 2].reshape(b, 1, t)
        return _term(
            op,
            a,
            v,
            params[:, :, 0].reshape(b, 1, t),
            params[:, :, 1].reshape(b, 1, t),
            self.operators,
        )

    def _normalize(self, phi, y, mask):
        b, n, t, k = self.batch, self.rows, self.terms, self.width
        anchor = phi[:, self.anchor_row : self.anchor_row + 1, :]
        delta = phi - anchor
        train = mask.reshape(1, n, 1)
        mean = (delta * train).sum(axis=1, keepdim=True) / self.ntrain
        centered = delta - mean
        scale = (
            (centered.square() * train).sum(axis=1, keepdim=True) / self.ntrain
        ).sqrt()
        scale = scale.eq(0).where(1.0, scale)
        normalized = centered / scale
        target = (y - self.ym).reshape(1, n, 1).expand(b, n, 1)
        design = normalized.cat(target, dim=2) * train
        if k > t + 1 or n < k:
            design = design.pad(((0, 0), (0, max(0, k - n)), (0, k - t - 1)))
        # Keep anchor and small shifted mean separate through the f32 readback.
        stats = anchor.cat(mean, scale, dim=2).reshape(b, 3 * t)
        return [design.contiguous(), stats.contiguous()]

    def _tensor_loss(self, phi, y, mask, coef):
        delta = phi - phi[:, self.anchor_row : self.anchor_row + 1, :]
        prediction = (delta * coef[:, None, : self.terms]).sum(axis=2) + coef[
            :, self.terms : self.terms + 1
        ]
        squared = (prediction - y.reshape(1, self.rows)).square()
        train = (squared * mask).sum(axis=1) / self.ntrain
        valid = (squared * (1 - mask)).sum(axis=1) / max(1, self.nvalid)
        return train.stack(valid, dim=1)

    def _custom_loss(self, phi, y, mask, coef):
        b, n, t, anchor, nt, nv = (
            self.batch,
            self.rows,
            self.terms,
            self.anchor_row,
            self.ntrain,
            max(1, self.nvalid),
        )

        def kernel(out, features, target, train, weights):
            out, features, target, train, weights = (
                v.flatten() for v in (out, features, target, train, weights)
            )
            c = UOp.range(out.ctx, b, 0)
            r = UOp.range(out.ctx, n, 1, AxisType.REDUCE)
            prediction = weights[c * (t + 1) + t]
            for j in range(t):
                delta = (
                    features[(c * n + r) * t + j] - features[(c * n + anchor) * t + j]
                )
                prediction = prediction + delta * weights[c * (t + 1) + j]
            squared = (prediction - target[r]) * (prediction - target[r])
            a = out[c * 2].store((squared * train[r]).sum(r) / nt)
            v = out[c * 2 + 1].store((squared * (1 - train[r])).sum(r) / nv)
            return a.group(v).end(c).sink(arg=KernelInfo(name="sym_family_losses"))

        return self._empty((b, 2)).custom_kernel(phi, y, mask, coef, fxn=kernel)[0]

    def _check_cosine_phase(self, descriptors):
        check_phases(self.feature_bounds, descriptors, self.npdtype)

    def score(self, descriptors, *, inspect=False):
        if self.rt is None:
            raise RuntimeError("family scorer is disposed")
        if descriptors.shape != (self.batch, self.terms, 5):
            raise ValueError("descriptor shape changed")
        if not set(descriptors[:, :, 2].ravel().astype(int)) <= set(self.operators):
            raise ValueError("operator outside compiled set")
        self._check_cosine_phase(descriptors)
        counters = self.rt.stats()
        start = time.perf_counter()
        self.indices.copy_from(descriptors[:, :, :3].astype(np.int32))
        self.params.copy_from(
            descriptors[:, :, 3:].astype(np.float32).astype(self.npdtype)
        )
        phi = self.feature.run([self.X, self.indices, self.params])
        design, stats = self.normalize.run([phi, self.y, self.mask])
        factors = self.factor.run([design])
        host_stats, host_factors = self.rt.Tensor.numpy_many([stats, factors])
        if not np.isfinite(host_stats).all() or not np.isfinite(host_factors).all():
            raise FloatingPointError("nonfinite device statistics or QR factors")
        coefficients = self.solve(host_factors, host_stats)
        # Evaluate original-unit coefficients with a shifted intercept. Casting a
        # large original bias to f32 would reintroduce cancellation at prediction.
        packed = coefficients.copy()
        packed[:, self.terms] += np.sum(
            coefficients[:, : self.terms] * host_stats[:, : self.terms].astype(float),
            axis=1,
        )
        self.coef.copy_from(packed.astype(self.npdtype))
        losses = self.loss.run([phi, self.y, self.mask, self.coef]).numpy()
        if not self.nvalid:
            losses[:, 1] = losses[:, 0]
        self.calls += 1
        # All GPU work is complete after loss readback. Retire temporary graph
        # owners while retaining the compiled stages and their resident buffers.
        if self.calls % 32 == 0:
            self.rt.collect()
        elapsed = (time.perf_counter() - start) * 1000
        if not np.isfinite(losses).all():
            raise FloatingPointError("nonfinite device losses")
        after = self.rt.stats()
        keys = [
            "launch_count",
            "buffer_read_count",
            "buffer_read_bytes",
            "buffer_write_count",
            "buffer_write_bytes",
            "buffer_copy_count",
            "buffer_copy_bytes",
            "runtime_cache_misses",
        ]
        traffic = {k: after[k] - counters[k] for k in keys}
        return {
            "coefficients": coefficients,
            "losses": losses,
            "ms": elapsed,
            "traffic": traffic,
            "features": phi.numpy() if inspect else None,
            "stats": host_stats if inspect else None,
        }

    def dispose(self):
        for stage in reversed(self.stages):
            stage.dispose()
        self.stages.clear()
        for tensor in self.outputs:
            tensor.dispose()
        self.outputs.clear()
        for name in ("X", "y", "mask", "indices", "params", "coef"):
            tensor = getattr(self, name, None)
            if tensor is not None:
                tensor.dispose()
                setattr(self, name, None)
        if self.rt is not None and self.owned_runtime:
            self.rt.dispose()
        self.rt = None


def check_phases(feature_bounds, descriptors, dtype):
    # Propagate conservative magnitude bounds through earlier-term sources.
    # SIN-based cosine lowering loses the phase shift at large magnitudes.
    bounds = np.empty(
        (descriptors.shape[0], len(feature_bounds) + descriptors.shape[1])
    )
    bounds[:, : len(feature_bounds)] = feature_bounds
    rows = np.arange(descriptors.shape[0])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for t in range(descriptors.shape[1]):
            d = descriptors[:, t]
            a = bounds[rows, d[:, 0].astype(int)]
            b = bounds[rows, d[:, 1].astype(int)]
            z = a * np.abs(d[:, 3]) + np.abs(d[:, 4])
            phase = z.astype(np.dtype(dtype))
            invalid = (~np.isfinite(phase)) | (np.spacing(phase) > 2e-6)
            if np.any((d[:, 2] == 5) & invalid):
                raise FloatingPointError(
                    "cosine phase exceeds the scorer precision budget; use C"
                )
            choices = [
                a + b,
                a + b,
                a * b,
                a / 1e-6,
                np.ones_like(a),
                np.ones_like(a),
                np.ones_like(a),
                np.maximum(abs(np.log(1e-6)), abs(np.log(z + 1e-6))),
                np.sqrt(z),
                np.full_like(a, np.exp(6)),
            ]
            bounds[:, len(feature_bounds) + t] = np.choose(d[:, 2].astype(int), choices)

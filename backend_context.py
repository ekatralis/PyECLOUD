import numpy as np
import numpy.random as np_random
import cupy as cp


class BackendContext(object):
    def __init__(self, use_gpu):
        self.use_gpu = bool(use_gpu)
        self.array_backend = cp if self.use_gpu else np
        self.random_backend = cp.random if self.use_gpu else np_random

    def to_numpy(self, values):
        if self.use_gpu:
            return cp.asnumpy(values)
        return np.asarray(values)

    def count_nonzero(self, values):
        return int(self.array_backend.count_nonzero(values))

    def sum_to_float(self, values):
        return float(self.array_backend.sum(values))

    def scalar_to_float(self, value):
        return float(value.item() if hasattr(value, 'item') else value)

    def any(self, values):
        return bool(self.array_backend.any(values).item())

    def repeat(self, values, repeats):
        if not self.use_gpu:
            return self.array_backend.repeat(values, repeats)

        if np.isscalar(repeats):
            return self.array_backend.repeat(values, int(repeats))

        counts = repeats
        if getattr(counts, "ndim", 0) == 0:
            return self.array_backend.repeat(values, int(self.scalar_to_float(counts)))

        counts_i64 = counts.astype(self.array_backend.int64, copy=False)
        total = int(self.scalar_to_float(self.array_backend.sum(counts_i64)))
        if total == 0:
            return values[:0].copy()

        offsets = self.array_backend.cumsum(counts_i64, dtype=self.array_backend.int64)
        take_idx = self.array_backend.searchsorted(
            offsets,
            self.array_backend.arange(total, dtype=self.array_backend.int64),
            side='right',
        )
        return values[take_idx]


def build_backend_context(use_gpu):
    return BackendContext(use_gpu)

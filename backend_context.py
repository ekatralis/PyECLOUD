import numpy as np
import numpy.random as np_random
import cupy as cp


class BackendContext(object):
    def __init__(self, use_gpu):
        self.use_gpu = bool(use_gpu)
        self.array_backend = cp if self.use_gpu else np
        self.random_backend = cp.random if self.use_gpu else np_random

    def count_nonzero(self, values):
        return int(self.array_backend.count_nonzero(values))

    def sum_to_float(self, values):
        return float(self.array_backend.sum(values))

    def scalar_to_float(self, value):
        return float(value.item() if hasattr(value, 'item') else value)

    def any(self, values):
        return bool(self.array_backend.any(values).item())


def build_backend_context(use_gpu):
    return BackendContext(use_gpu)

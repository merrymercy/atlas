from typing import Any, Callable, Collection

import numpy as np

class NumpyChecker:
    @staticmethod
    def check(v1: Any, v2: Any):
        if isinstance(v1, (tuple, list)):
            if type(v1) != type(v2) or len(v1) != len(v2):
                return False
            return all([NumpyChecker.check(x, y) for x, y in zip(v1, v2)])
        if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
            try:
                return v1 == v2
            except ValueError:
                return False
        if v1.dtype != v2.dtype:
            return False
        if v1.shape != v2.shape:
            return False
        return np.allclose(v1, v2)

from typing import Any, Callable, Collection

import numpy as np

def convert_np_type(x):
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    elif isinstance(x, (np.float32, np.float64)):
        return float(x)
    else:
        return x


class NumpyChecker:
    @staticmethod
    def check(v1: Any, v2: Any):
        if isinstance(v1, (tuple, list)) or isinstance(v2, (tuple, list)):
            if len(v1) != len(v2):
                return False
            return all([NumpyChecker.check(x, y) for x, y in zip(v1, v2)])
        if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
            v1 = convert_np_type(v1)
            v2 = convert_np_type(v2)

            if type(v1) != type(v2):
                return False
            try:
                if isinstance(v1, float):
                    return np.allclose(v1, v2)
                return v1 == v2
            except ValueError:
                return False
        if v1.dtype != v2.dtype:
            return False
        if v1.shape != v2.shape:
            return False
        return np.allclose(v1, v2)

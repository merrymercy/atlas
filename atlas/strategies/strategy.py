import collections
from abc import abstractmethod, ABC
from typing import Callable, Tuple, Dict, Any, Optional, Set


def operator(func):
    setattr(func, "_is_generator_op", True)
    return func


def is_operator(func):
    return getattr(func, "_is_generator_op", False)


class Strategy(ABC):
    def __init__(self):
        self.op_cnt: Dict[str, int] = collections.defaultdict(int)
        self.known_ops: Set[str] = {k for k in dir(self) if is_operator(getattr(self, k))}

    def init(self):
        pass

    def finish(self):
        pass

    def init_run(self):
        pass

    def finish_run(self):
        pass

    @abstractmethod
    def is_finished(self):
        pass

    def get_known_ops(self):
        return self.known_ops

    def process_op(self, op_kind: str, op_id: Optional[str] = None) -> Tuple[str, Callable]:
        if op_id is None:
            self.op_cnt[op_kind] += 1
            op_id = str(self.op_cnt[op_kind])

        op_name: str = op_kind + "_" + str(op_id)
        return op_name, self.make_op(op_kind, op_id)

    @abstractmethod
    def make_op(self, op_kind: str, op_id: str) -> Callable:
        pass

import collections
from abc import abstractmethod, ABC
from typing import Callable, Optional, Set

from atlas.models.core import GeneratorModel
from atlas.operators import OpInfo, is_operator, is_method, get_attrs, OpResolvable, resolve_operator, \
    find_known_operators, find_known_methods


class Strategy(OpResolvable):
    def __init__(self):
        self.known_ops = find_known_operators(self)
        self.known_methods = find_known_methods(self)

    def get_op_handler(self, op_info: OpInfo):
        return resolve_operator(self.known_ops, op_info)

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

    def get_known_methods(self):
        return self.known_methods

    def generic_call(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                     **kwargs):
        pass


class IteratorBasedStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()
        self.model: Optional[GeneratorModel] = None

    def set_model(self, model: Optional[GeneratorModel]):
        self.model = model

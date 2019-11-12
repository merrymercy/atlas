import random
import itertools
from typing import Collection, Dict, Optional, Any, List, Callable, Iterator

import numpy as np

from ...exceptions import ExceptionAsContinue, StepTermination
from ...operators import OpInfo, operator
from ...models import IndependentOperatorsModel, TrainableModel, GeneratorModel
from ...models.tensorflow.graphs.operators import GGNN, SelectGGNN, SelectFixedGGNN,\
    SequenceGGNN, SequenceFixedGGNN, OrderedSubsetGGNN, SubsetGGNN 
from ..utils import dump_encodings
from ...strategy import Strategy
from ...strategies import RandStrategy, DfsStrategy
from .encoders import NumpyGraphEncoder


def SelectExternal(domain: Any, dtype=None, preds: List[Callable] = None, **kwargs):
    if preds is None:
        preds = []

    if dtype:
        domain = [x for x in domain if isinstance(x, dtype) and all(p(x) for p in preds)]
    else:
        domain = [x for x in domain if all(p(x) for p in preds)]
    unused_intermediates: Set[int] = kwargs.get('unused_intermediates', None)
    if unused_intermediates is not None:
        domain1 = [x for x in domain if id(x) in unused_intermediates]
        domain2 = [x for x in domain if id(x) not in unused_intermediates]
        yield from domain1   # todo(lmzheng): investigate this
        yield from domain2
    else:
        yield from (x for x in domain if isinstance(x, dtype))

class NumpyRandomStrategy(RandStrategy):
    def __init__(self, model: GeneratorModel=None, max_iter: Optional[int] = None):
        self.max_iter = max_iter
        self.iter_ct = 0
        self.model = model

        super().__init__()

    def is_finished(self):
        if self.max_iter is not None:
            return self.iter_ct > self.max_iter or super().is_finished()
        else:
            return super().is_finished()

    def init(self):
        self.iter_ct = 0

    def finish_run(self):
        self.iter_ct += 1
 
    @operator
    def SelectExternal(self, domain, dtype=None, preds: List[Callable] = None, **kwargs):
        if preds is None:
            preds = []

        if dtype:
            domain = [x for x in domain if isinstance(x, dtype) and all(p(x) for p in preds)]
        else:
            domain = [x for x in domain if all(p(x) for p in preds)]

        unused_intermediates: Set[int] = kwargs.get('unused_intermediates', None)
        if unused_intermediates is not None:
            weight = np.array([5 if id(x) in unused_intermediates else 1 for x in domain])
            weight = weight / np.sum(weight)
            return domain[np.random.choice(len(domain), p=weight)]
        else:
            return domain[np.random.choice(domain)]

    @operator
    def Select(self, domain: Any, context=None, **kwargs):
        if self.model:
            val, probs = self.model.infer(domain=domain, context=context, 
                                         return_prob=True, **kwargs)
            probs = probs / np.sum(probs)
            return val[np.random.choice(range(len(val)), p=probs)]
        return super().Select(domain, context=context, **kwargs)

    @operator
    def SelectFixed(self, domain: Any, context=None, **kwargs):
        return self.Select(domain, context, **kwargs)

    @operator
    def Subset(self, domain: Any, context: Any = None, lengths: Collection[int] = None,
               include_empty: bool = False, **kwargs):
        if self.model:
            val, probs = self.model.infer(domain=domain, context=context,
                                          lengths=lengths, include_empty=include_empty,
                                          return_prob=True, **kwargs)
            probs = probs / np.sum(probs)
            return val[np.random.choice(range(len(val)), p=probs)]

        return super().Subset(domain, context=context, lengths=lengths,
                              include_empty=include_empty, **kwargs)

    @operator
    def OrderedSubset(self, domain: Any, context: Any = None,
                      lengths: Collection[int] = None, include_empty: bool = False, **kwargs):
        if self.model:
            val, probs = self.model.infer(domain=domain, context=context,
                                          lengths=lengths, include_empty=include_empty,
                                          return_prob=True, **kwargs)
            probs = probs / np.sum(probs)
            return val[np.random.choice(range(len(val)), p=probs)]

        return super().OrderedSubset(domain, context=context, lengths=lengths,
                                     include_empty=include_empty, **kwargs)

    @operator
    def Sequence(self, domain: Any, context: Any = None, max_len: int = None,
                 lengths: Collection[int] = None, **kwargs):
        if self.model:
            val, probs = self.model.infer(domain=domain, context=context,
                                          max_len=max_len, return_prob=True,
                                          **kwargs)
            probs = probs / np.sum(probs)
            return val[np.random.choice(range(len(val)), p=probs)]

        return super().Sequence(domain, context=context, max_len=max_len,
                                lengths=lengths, **kwargs)

    @operator
    def SequenceFixed(self, domain: Any, context: Any = None, max_len: int = None,
                      lengths: Collection[int] = None, **kwargs):
        return self.Sequence(domain, context, max_len, lengths, **kwargs)


class NumpyDfsStrategy(DfsStrategy):
    def __init__(self, random_filter: Callable = None, 
                 operator_iterator_bound: int = None,
                 max_iter: Optional[int] = None):
        self.random_filter = random_filter
        self.max_iter = max_iter
        self.iter_ct = 0
        super().__init__(operator_iterator_bound=operator_iterator_bound)

    def init(self):
        self.iter_ct = 0
        super().init()

    def finish_run(self):
        self.iter_ct += 1
        super().finish_run()

    def is_finished(self):
        if self.max_iter is not None:
            return self.iter_ct > self.max_iter or super().is_finished()
        else:
            return super().is_finished()

    @operator
    def SelectExternal(self, domain, dtype=None, preds: List[Callable] = None, **kwargs):
        yield from SelectExternal(domain, dtype, preds, **kwargs)

    def generic_call(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None,
                     **kwargs):
        t = self.call_id
        self.call_id += 1

        if t not in self.op_iter_map:
            try:
                iterator = None
                if self.model is not None:
                    try:
                        iterator = self.model.infer(domain=domain, context=context, op_info=op_info, **kwargs)
                    except NotImplementedError:
                        pass

                if iterator is None:
                    iterator = handler(self, domain=domain, context=context, op_info=op_info, **kwargs)

                if self.random_filter and self.random_filter(op_info.sid):
                    iterator = list(iterator)
                    random.shuffle(iterator)

                if self.operator_iterator_bound:
                    op_iter = itertools.islice(iter(iterator), self.operator_iterator_bound)
                else:
                    op_iter = iter(iterator)

                self.op_iter_map[t] = op_iter
                val = self.val_map[t] = next(op_iter)

            except StopIteration:
                #  Operator received an empty domain
                raise ExceptionAsContinue

        else:
            val = self.val_map[t]

        return val


class NumpyBeamSearchStrategy(Strategy):
    def __init__(self, model: GeneratorModel = None, 
                 max_iter: Optional[int] = None, 
                 operator_iterator_bound: int = 10, 
                 beam_size: int = 100):
        self.model = model
        self.max_iter = max_iter
        self.iter_ct = 0
        self.operator_iterator_bound = operator_iterator_bound
        self.beam_size = beam_size

        self.current_idx = 0
        self.current_trace = []

        super().__init__()

    def is_finished(self):
        return False

    def init(self):
        self.iter_ct = 0

    def reset_trace(self, trace):
        self.current_trace = trace
        self.current_idx = 0

    def generic_call(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None, **kwargs):
        if self.current_idx < len(self.current_trace):
            self.current_idx += 1
            return self.current_trace[self.current_idx - 1]

        return handler(self, domain=domain, context=context, op_info=op_info, **kwargs)

    @operator
    def SelectExternal(self, domain, dtype=None, preds: List[Callable] = None, **kwargs):
        if preds is None:
            preds = []
        domain = [x for x in domain if isinstance(x, dtype) and all(p(x) for p in preds)]

        unused_intermediates: Set[int] = kwargs.get('unused_intermediates', None)
        if unused_intermediates is not None:
            weight = np.array([5 if id(x) in unused_intermediates else 1 for x in domain])
            weight = weight / np.sum(weight)
            raise StepTermination(domain[:self.operator_iterator_bound],
                                  weight[:self.operator_iterator_bound])
        else:
            weight = np.ones((len(domain),)) / len(domain)
            raise StepTermination(domain[:self.operator_iterator_bound],
                                  weight[:self.operator_iterator_bound])

    @operator
    def Select(self, domain: Any, context=None, **kwargs):
        val, probs = self.model.infer(domain=domain, context=context, 
                                     return_prob=True, **kwargs)
        probs = probs / np.sum(probs)
        raise StepTermination(val[:self.operator_iterator_bound],
                              probs[:self.operator_iterator_bound])

    @operator
    def SelectFixed(self, domain: Any, context=None, **kwargs):
        return self.Select(domain, context, **kwargs)

    @operator
    def Subset(self, domain: Any, context: Any = None, lengths: Collection[int] = None,
               include_empty: bool = False, **kwargs):
        val, probs = self.model.infer(domain=domain, context=context,
                                      lengths=lengths, include_empty=include_empty,
                                      return_prob=True, **kwargs)
        probs = probs / np.sum(probs)
        raise StepTermination(val[:self.operator_iterator_bound],
                              probs[:self.operator_iterator_bound])

    @operator
    def OrderedSubset(self, domain: Any, context: Any = None,
                      lengths: Collection[int] = None, include_empty: bool = False, **kwargs):
        val, probs = self.model.infer(domain=domain, context=context,
                                      lengths=lengths, include_empty=include_empty,
                                      return_prob=True, **kwargs)
        probs = probs / np.sum(probs)
        raise StepTermination(val[:self.operator_iterator_bound],
                              probs[:self.operator_iterator_bound])

    @operator
    def Sequence(self, domain: Any, context: Any = None, max_len: int = None,
                 lengths: Collection[int] = None, **kwargs):
        a = self.model.infer(domain=domain, context=context, max_len=max_len, return_prob=True, **kwargs)
        val, probs = a
        probs = probs / np.sum(probs)
        raise StepTermination(val[:self.operator_iterator_bound],
                              probs[:self.operator_iterator_bound])

    @operator
    def SequenceFixed(self, domain: Any, context: Any = None, max_len: int = None,
                      lengths: Collection[int] = None, **kwargs):
        return self.Sequence(domain, context, max_len, lengths, **kwargs)



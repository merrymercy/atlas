from collections import namedtuple
from typing import Dict, Any, Callable, Collection, Optional, Iterator, Tuple

from atlas.exceptions import ExceptionAsContinue
from atlas.operators import OpInfo, operator
from atlas.tracing import OpTrace, GeneratorTrace
from atlas.strategies.strategy import Strategy, IteratorBasedStrategy

class FastReplayStrategy(Strategy):
    def __init__(self, trace):
        self.trace = iter(trace)

    def generic_call(self, *args, **kwargs):
        return next(self.trace)

class ParallelDfsStrategy(IteratorBasedStrategy):
    def __init__(self):
        super().__init__()

        self.iter_map = {}         # call_id -> iterator
        self.val_map = {}          # call_id -> value
        self.dependency_map = {}   # call_id -> bool
        self.domain_map = {}       # call_id -> domain
        self.op_info_map = {}      # call_id -> op_info
        self.sid_map = {}          # call_id -> str
        self.call_id = 0

        self.finished = False

        # shared 
        self.trace_batch = []
        self.visited = set()

    def init_run(self):
        self.call_id = 0
        self.stop_at = None

    def finish_run(self):
        # delete enumerated iterators
        for t in range(self.call_id - 1, self.stop_at, -1):
            del self.iter_map[t]

        # find a updateable iterator
        for t in range(self.stop_at, -1, -1):
            try:
                self.val_map[t] = next(self.iter_map[t])
                return
            except StopIteration:
                del self.iter_map[t]
                continue

        self.finished = True

    def is_finished(self):
        return self.finished

    def get_trace_batch(self):
        self.trace_batch = []
        self.visited = set()
        self.op_trace_stack = []

        self.simulate_dfs(0)

        if self.stop_at is None:
            self.stop_at = 0
            self.finished = True

        return self.trace_batch

    def simulate_dfs(self, call_id):
        if call_id == self.call_id:
            self.trace_batch.append(list(self.op_trace_stack))
        else:
            if call_id not in self.visited:   # reuse once
                self.visited.add(call_id)

                self.op_trace_stack.append(self.val_map[call_id])
                self.simulate_dfs(call_id + 1)
                self.op_trace_stack.pop()
                if self.stop_at is not None:
                    return

            if self.dependency_map[call_id]:
                self.stop_at = call_id
                return

            for v in self.iter_map[call_id]:
                self.op_trace_stack.append(v)
                self.simulate_dfs(call_id + 1)
                self.op_trace_stack.pop()
                if self.stop_at is not None:
                    return

            self.iter_map[call_id] = iter(self.domain_map[call_id])

    def generic_call(self, domain=None, context=None, op_info: OpInfo = None, handler: Optional[Callable] = None, **kwargs):
        t = self.call_id
        self.call_id += 1

        if t not in self.iter_map:
            try:
                iterator = handler(self, domain=domain, context=context, op_info=op_info, **kwargs)
                op_iter = iter(iterator)

                self.iter_map[t] = op_iter
                self.val_map[t] = val = next(op_iter)
                self.dependency_map[t] = kwargs.get('dependency', False)

                self.op_info_map[t] = op_info
                self.domain_map[t] = domain

            except StopIteration: #  Operator received an empty domain
                raise ExceptionAsContinue
        else:
            val = self.val_map[t]

        return val

    @operator
    def Select(self, domain: Any, context: Any = None, **kwargs):
        return domain


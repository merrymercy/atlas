import time
import pickle
#import cloudpickle

import numpy as np
import ray

import atlas
from atlas import generator
from atlas.operators import OpInfo
from atlas.strategies import ParallelDfsStrategy
from atlas.synthesis.numpy.api import gen_repeat

print(type(OpInfo(sid='/initial_extractor/Select@@1', gen_name='initial_extractor', op_type='Select',
                        index=1, gen_group=None, uid=None, tags=None)))


@atlas.generator
def initial_extractor(name):
    first = Select(name)
    last = Select(name)

    return f"{first}.{last}"

def sequential():
    initial_extractor.set_default_strategy('dfs')
    result = set()

    for ct, (val, trace) in enumerate(initial_extractor.generate("Alan").with_tracing()):
        result.add(val)

    return result

def parallel():
    initial_extractor.set_default_strategy(ParallelDfsStrategy())

    result = set()
    for ct, (val, trace) in enumerate(initial_extractor.generate("Alan").with_tracing().with_batch(32)):
        result.add(val)

    return result

def measure(func, args=[], kwargs={}, n=1):
    ta = time.time()
    for i in range(n):
        func(*args, **kwargs)
    tb = time.time()
    return (tb - ta) / n


@ray.remote
def call_func(func, args):
    return func(*args)

if __name__ == "__main__":
    ray.init(num_cpus=1)

    res1 = sequential()
    res2 = parallel()

    assert res1 == res2
    print(len(res1), len(res2))

    #print(f"Sequential : {measure(sequential)}")
    #print(f"Parallel   : {measure(parallel)}")


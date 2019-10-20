import time
import pickle

import numpy as np
import ray

import atlas
from atlas import generator
from atlas.strategies import ParallelDfsStrategy
from atlas.synthesis.numpy.api import gen_repeat


@atlas.generator
def initial_extractor(name):
    first = Select(name)
    last = Select(name)

    return f"{first}.{last}"

def sequential():
    initial_extractor.set_default_strategy('dfs')
    result = set()

    for ct, (val, trace) in enumerate(initial_extractor.generate("Alan Turing").with_tracing()):
        result.add(val)

    return result

def parallel():
    initial_extractor.set_default_strategy(ParallelDfsStrategy())

    result = set()
    for ct, (val, trace) in enumerate(initial_extractor.generate("Alan Turing").with_tracing().with_batch(32)):
        result.add(val)

    return result

def measure(func, args=[], kwargs={}, n=1):
    ta = time.time()
    for i in range(n):
        func(*args, **kwargs)
    tb = time.time()
    return (tb - ta) / n

def measure_initial_extractor():
    res1 = sequential()

    res2 = parallel()

    print(res2)

    assert res1 == res2
    print(len(res1))
    print(f"Sequential : {measure(sequential)}")
    print(f"Parallel   : {measure(parallel)}")

@atlas.generator
def numpy_test(inputs, output):
    val, prog = gen_repeat(inputs, output)
    inputs = inputs + [val]
    val, prog = gen_repeat(inputs, output)
    inputs = inputs + [val]
    val, prog = gen_repeat(inputs, output)

    return val, prog

def sequential_numpy():
    inputs = [np.random.randn(3, 4, 5)]
    output = None

    ct = 0
    for v, trace in numpy_test.generate(inputs, output).with_strategy('dfs')\
                              .with_tracing():
        ct += 1

    #print(ct)


def parallel_numpy():
    inputs = [np.random.randn(3, 4, 5)]
    output = None

    ct = 0
    for v, trace in numpy_test.generate(inputs, output).with_strategy('parallel-dfs')\
                              .with_tracing().with_batch():
        ct += 1

    #print(ct)

@ray.remote
def test_func(arg_1, arg_2):
    None


if __name__ == "__main__":
    config = ray.init(num_cpus=24)
    print("==================")
    print(config)
    print("==================")

    print(f"Parallel   : {measure(parallel_numpy)}")
    ray.timeline(filename='timeline.tracing')


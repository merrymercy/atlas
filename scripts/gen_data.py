import itertools, traceback, warnings, pickle, random, argparse
import numpy as np

import atlas
from atlas.synthesis.numpy import NumpyChecker, NumpyGraphEncoder, NumpyGeneratorModel, \
        numpy_sequence_generator

from stackoverflow_cases import cases

numpy_apis =  {gen.name: gen for gen in atlas.utils.get_group_by_name('numpy')}

def gen_sequence(seqs=[]):
    # len = 1
    return [['repeat']] #, 'dstack', 'reshape']]

    for name in numpy_apis:
        seqs.append([name])

    return seqs

    # len = 2
    create = ['arnage', 'eye', 'identity', 'ones', 'full']  # 'ones_like', 'full_like'
    shape = ['reshape', 'ravel', 'ndarray.flatten', 'moveaxis', 'rollaxis', 'swapaxes', 'transpose', 'expand_dims', 'squeeze']
    stack = ['concatenate', 'stack', 'column_stack', 'dstack', 'hstack', 'vstack']
    split = ['split', 'array_split', 'dsplit', 'hsplit', 'vsplit']
    repeat = ['repeat', 'tile']

    def gen_cross(a, b):
        for x in itertools.product(a, b):
            seq.append(x)

    gen_cross(create, shape)
    gen_cross(shape, stack)
    gen_cross(stock, split)
    gen_cross(split, repeat)
    # len = 3

    return seqs

def gen_sequence_cheat(seqs=[]):
    for case in cases:
        all_covered = True

        for name in case.func_sequence:
            if name not in numpy_apis:
                print(f"{name} is missing for case {case.no}!")
                all_covered = False

        if all_covered:
            seqs.append(case.func_sequence)

    return seqs


def get_used_functions(seqs):
    used_funcs = set()
    for x in seqs:
        used_funcs = used_funcs.union(x)

    return list(used_funcs)


def random_ndarray(max_dim=4, max_per_dim=4, dtypes=[np.float]):
    ndim = np.random.randint(1, max_dim)
    shape = []
    for i in range(ndim):
        shape.append(np.random.randint(1, max_per_dim))

    return np.random.randn(*shape).astype(random.choice(dtypes))

def random_vectors(max_number=4, max_length=5, dtypes=[np.float]):
    number = np.random.randint(1, max_number)
    length = np.random.randint(1, max_length)
    dtype = np.random.choice(dtypes)

    return [np.random.randn(length).astype(dtype) for _ in range(number)]

def infinite_list(x):
    while True:
        yield x

MAX_ERROR_RETRY = 8

def gen_traces(seqs, size_per_seq, seq_per_input, funcs, strategy):
    traces = []
    traces_args = []
    test_cases = []

    numpy_sequence_generator.set_default_strategy(strategy)

    for seq in seqs:
        i = 0
        error_ct = 0
        n_inputs = 0
        while i < size_per_seq and error_ct < MAX_ERROR_RETRY:
            inputs = [random_ndarray()]
            #inputs = random_vectors()

            ct = 0
            for (output, prog), trace in numpy_sequence_generator.generate(inputs, None, funcs=funcs)\
                                .with_tracing().with_replay({"func_seq" : infinite_list(seq)}):
                traces.append(trace)
                traces_args.append((inputs, output))
                test_cases.append((inputs, output))
                ct += 1
                if i + ct >= size_per_seq or ct > seq_per_input:
                    break
            n_inputs += 1
            if ct == 0:  # no traces gathered, it menas some errors happened
                error_ct += 1
            else:
                error_ct = 0
            i += ct

        if error_ct >= MAX_ERROR_RETRY:
            print(f"Skip error sequence {seq}")
        print(f"{seq}: Enumerate space for {n_inputs} inputs, average space size: {i // n_inputs}")

    traces = [numpy_sequence_generator.generate(*args).with_replay(trace).with_tracing().first()[1] 
              for trace, args in zip(traces, traces_args)]
    return traces, test_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-only', type=int, default=1)
    parser.add_argument('--train-size-per-seq', type=int, default=10000)
    parser.add_argument('--test-size-per-seq', type=int, default=1000)
    parser.add_argument('--seq-per-input', type=int, default=20)

    args = parser.parse_args()

    seqs = gen_sequence()
    funcs = get_used_functions(seqs)
    
    print(f"Load {len(seqs)} sequences :  {seqs}")
    print('============')

    if args.n_only == 0:
        args.n_only = len(seqs)

    train_data, _ = gen_traces(seqs[:args.n_only], args.train_size_per_seq, args.seq_per_input,
            funcs, 'dfs')
    test_data, test_cases = gen_traces(seqs[:args.n_only], args.test_size_per_seq, args.seq_per_input,
            funcs, 'dfs')

    print(len(train_data), len(test_data))
    random.shuffle(train_data)
    random.shuffle(test_data)

    pickle.dump(train_data, open('train_traces.pkl', 'wb'))
    pickle.dump(test_data, open('test_traces.pkl', 'wb'))
    pickle.dump(test_cases, open('test_cases.pkl', 'wb'))


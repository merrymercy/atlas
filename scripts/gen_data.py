import itertools, traceback, warnings, pickle, random, argparse
import shutil
from collections import namedtuple
import random

from tqdm import tqdm
import numpy as np

import atlas
from atlas.models import IndependentOperatorsModel
from atlas.synthesis.numpy import NumpyChecker, NumpyGraphEncoder, NumpyGeneratorModel, \
        numpy_sequence_generator, NumpyRandomStrategy, NumpyDfsStrategy

from stackoverflow_cases import cases

numpy_apis =  {gen.name: gen for gen in atlas.utils.get_group_by_name('numpy')}

TestSequence = namedtuple("TestSequnce", ['used_api', 'gen_inputs'])

MIN_DIM = 1
MAX_DIM = 4
MAX_PER_DIM = 4


def random_ndarray(min_dim=MIN_DIM, max_dim=MAX_DIM, max_per_dim=MAX_PER_DIM, dtypes=[np.float]):
    ndim = np.random.randint(min_dim, max_dim + 1)
    shape = []
    for i in range(ndim):
        shape.append(np.random.randint(1, max_per_dim + 1))

    return np.random.randn(*shape).astype(random.choice(dtypes))

def random_ndarrays(number=1, **kwargs):
    ret = []
    if isinstance(number, (list, tuple)):
        number = np.random.randint(number[0], number[1] + 1)

    for i in range(number):
        ret.append(random_ndarray(**kwargs))
    return ret

def random_ndarrays_with_duplicate(number=1, **kwargs):
    ret = random_ndarrays(number, **kwargs)
    for i in range(len(ret)):
        axis = np.random.choice(range(ret[i].ndim))
        pos = np.random.choice(range(ret[i].shape[axis]))
        value = np.take(ret[i], np.random.choice(range(ret[i].shape[axis])), axis=axis)
        dup_number = np.random.choice(range(1, 3))

        value = np.stack((value, value))

        ret[i] = np.insert(ret[i], pos, value, axis=axis)

    return ret

def random_ndarrays_with_nan(number=1, **kwargs):
    ret = random_ndarrays(number, **kwargs)
    probability = 0.2

    for i in range(len(ret)):
        view = ret[i].view().reshape((-1,))
        for j in range(ret[i].size):
            if np.random.random() < probability:
                view[j] = np.nan

    return ret

def random_ndarrays_with_scalar(number=1, **kwargs):
    ret = random_ndarrays(number, **kwargs)
    scalar_array = np.random.randn(1)
    ret.append(scalar_array)

    return ret

def random_ndarrays_for_matmul(max_per_dim=MAX_PER_DIM):
    m, n, k = [np.random.choice(np.arange(1, max_per_dim+1)) for _ in range(3)]
    lhs = np.random.randn(m, k)
    rhs = np.random.randn(k, n)
    return [lhs, rhs]

def random_ndarrays_for_hstack():
    ndim = np.random.choice(range(1, 4))
    a = random_ndarray(min_dim=ndim, max_dim=ndim)

    if ndim == 1:
        b = random_ndarray(min_dim=ndim, max_dim=ndim)
    else:
        shape_b = list(a.shape)
        shape_b[1] = np.random.choice(range(1, MAX_PER_DIM + 1))
        b = np.random.randn(*shape_b)

    return [a, b]

def random_vectors(number=2, min_length=2, max_length=6,
                   dtypes=[np.float], same_length=False):
    if isinstance(number, (list, tuple)):
        number = np.random.randint(number[0], number[1] + 1)
    dtype = np.random.choice(dtypes)

    if same_length:
        length = np.random.randint(min_length, max_length + 1)
        return [np.random.randn(length).astype(dtype) for _ in range(number)]
    else:
        return [np.random.randn(np.random.randint(min_length, max_length + 1))
                for _ in range(number)]

def random_vectors_with_nan(number=1, nan_prob=0.2, **kwargs):
    ret = random_vectors(number, **kwargs)

    for i in range(len(ret)):
        view = ret[i].view().reshape((-1,))
        for j in range(ret[i].size):
            if np.random.random() < nan_prob:
                view[j] = np.nan

    return ret

def random_vectors_with_index(number=1, **kwargs):
    ret = random_vectors(number, **kwargs)

    arr = ret[np.random.choice(range(len(ret)))]
    index_mask = np.random.randn(*arr.shape) < 0.5
    index = np.arange(arr.size)[index_mask]

    ret.append(index)
    return ret

def random_vectors_with_subvector(number=1, **kwargs):
    ret = random_vectors(number, **kwargs)

    subvectors = []
    for arr in ret:
        index_mask = np.random.randn(*arr.shape) < 0.5
        sub_arr = arr[index_mask]
        subvectors.append(sub_arr)

    ret.extend(subvectors)
    return ret

def random_vectors_with_a_bool_array(number=1, **kwargs):
    ret = random_vectors(number, **kwargs)

    arr = ret[0]
    bool_array = np.random.randn(*arr.shape) < 0.5

    ret.append(bool_array)
    return ret

def random_index_and_shape():
    n_dim = np.random.choice(range(MIN_DIM, MAX_DIM + 1))
    shape = []
    for i in range(n_dim):
        shape.append(np.random.choice(range(1, MAX_PER_DIM+1)))

    return shape


def gen_sequence(seqs=[]):
    return [
        #TestSequence(['unique'], lambda : random_ndarrays_with_duplicate()),
        TestSequence(['meshgrid'], lambda : random_vectors()),
        TestSequence(['dstack'], lambda : random_vectors(same_length=True)),
        TestSequence(['reshape'], lambda : random_ndarrays()),
        TestSequence(['ndarray.__sub__'], lambda : random_vectors(same_length=True)),
        TestSequence(['linalg.norm'], lambda : random_ndarrays()),
        #TestSequence(['matmul'], lambda : random_ndarrays_for_matmul()),
        #TestSequence(['sum'], lambda : random_ndarrays()),
        TestSequence(['isnan'], lambda : random_vectors_with_nan()),
        TestSequence(['logical_not'], lambda : random_ndarrays()),
        TestSequence(['compress'], lambda : random_vectors_with_a_bool_array()),

        TestSequence(['ndarray.__div__'], lambda : random_vectors(same_length=True)),
        TestSequence(['swapaxes'], lambda : random_ndarrays(min_dim=2)),
        # Todo(lmzheng): fix this later
        #TestSequence(['unravel_index'], lambda : random_index_and_shape()),
        TestSequence(['argsort'], lambda : random_ndarrays()),
        TestSequence(['flip'], lambda : random_ndarrays()),
        TestSequence(['ndarray.__getitem__'], lambda : random_ndarrays()),
        TestSequence(['take'], lambda : random_ndarrays()),
        TestSequence(['zeros'], lambda : []),
        TestSequence(['hstack'], lambda : random_ndarrays_for_hstack()),
        TestSequence(['transpose'], lambda : random_ndarrays()),

        # No. 3
        TestSequence(['unique'], lambda : random_ndarrays_with_duplicate()),
        # No. 4
        TestSequence(['meshgrid', 'dstack', 'reshape'], lambda : random_vectors()),
        # No. 6
        TestSequence(['ndarray.__sub__', 'linalg.norm'], lambda : random_vectors(same_length=True)),
        # No. 8
        TestSequence(['matmul'], lambda : random_ndarrays_for_matmul()),
        # No. 9
        TestSequence(['sum'], lambda : random_ndarrays()),
        # No. 10
        TestSequence(['isnan', 'logical_not', 'compress'], lambda: random_vectors_with_nan()),
        # No. 12
        TestSequence(['linalg.norm', 'ndarray.__div__'], lambda : random_ndarrays()),
        # No. 13
        TestSequence(['ndarray.flatten'], lambda : random_ndarrays()),
        # No. 14
        TestSequence(['vstack'], lambda : random_vectors(same_length=True)),
        # No. 15
        TestSequence(['reshape', 'swapaxes'], lambda : random_ndarrays()),
        # No. 16
        TestSequence(['argmax', 'unravel_index'], lambda : random_ndarrays()),
        # No. 17
        TestSequence(['vstack', 'reshape'], lambda : random_vectors(same_length=True)),
        # No. 21
        TestSequence(['argsort', 'flip', 'ndarray.__getitem__'], lambda : random_vectors(1)),
        # No. 22
        TestSequence(['take', 'argsort', 'take'], lambda : random_ndarrays()),
        # No. 25
        TestSequence(['reshape', 'swapaxes', 'reshape'], lambda: random_ndarrays()),
        # No. 26  hard to generate testcases
        TestSequence(['zeros', 'hstack'], lambda: random_ndarrays()),
        # No. 29
        TestSequence(['unique', 'vstack', 'transpose'], lambda: random_vectors(1)),
        # No. 32   duplication of No.13
        # No. 33   duplication of No.14
        ## No. 34  hard to generate testcases
        #TestSequence(['repeat', 'reshape', 'hstack'], lambda: random_ndarrays_with_scalar()),
        # No. 35
        TestSequence(['delete'], lambda: random_vectors_with_index()),
        # No. 36
        TestSequence(['argmax'], lambda: random_ndarrays()),
        # No. 38
        TestSequence(['searchsorted'], lambda: random_vectors_with_subvector(1)),
        # No. 39
        TestSequence(['minimum'], lambda: random_ndarrays_with_scalar()),
    ]

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
    ret = []
    for funcs, _ in seqs:
        for func in funcs:
            if func not in used_funcs:
                used_funcs.add(func)
                ret.append(func)
    return ret


def infinite_list(x):
    while True:
        yield x

MAX_ERROR_RETRY = 8

def gen_traces(seqs, funcs, size_per_seq, max_seq_per_input):
    random_stra = NumpyRandomStrategy(max_iter=max_seq_per_input * 3)
    dfs_stra = NumpyDfsStrategy()

    traces = []
    traces_args = []
    test_cases = []


    for seq, input_generator in seqs:
        current_stra = dfs_stra
        numpy_sequence_generator.set_default_strategy(current_stra)
        i = 0
        error_ct = 0
        n_inputs = 0
        while i < size_per_seq and error_ct < MAX_ERROR_RETRY:
            inputs = input_generator()

            ct = 0
            for (output, prog), trace in numpy_sequence_generator.generate(inputs, None, funcs=funcs)\
                                .with_tracing().with_replay({"func_seq" : infinite_list(seq)}):
                traces.append(trace)
                traces_args.append((inputs, output, funcs))
                test_cases.append((inputs, output))
                ct += 1
                if i + ct >= size_per_seq or ct > max_seq_per_input:
                    break

            if ct == 0:  # no traces gathered, it menas some errors happened
                error_ct += 1
            else:
                error_ct = 0

            # switch to random stra if the search space for a single input is too large
            if ct > max_seq_per_input and current_stra == dfs_stra:
                current_stra = random_stra
                numpy_sequence_generator.set_default_strategy(current_stra)

            n_inputs += 1
            i += ct

        if error_ct >= MAX_ERROR_RETRY:
            print(f"Skip error sequence {seq}")
        print(f"{seq}: Enumerate space for {n_inputs} inputs, average space size: {i / n_inputs: .2f}")

    traces = [numpy_sequence_generator.generate(*args).with_replay(trace).with_tracing().first()[1] 
              for trace, args in zip(traces, traces_args)]
    return traces, test_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-only', type=int, default=0)
    parser.add_argument('--train-size-per-seq', type=int, default=10000)
    parser.add_argument('--test-size-per-seq', type=int, default=1000)
    parser.add_argument('--max-seq-per-input', type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    seqs = gen_sequence()
    funcs = get_used_functions(seqs)
    
    print(f"Load {len(seqs)} sequences :  {[x.used_api for x in seqs]}")
    print('============')

    if args.n_only == 0:
        args.n_only = len(seqs)

    train_data, _ = gen_traces(seqs[:args.n_only], funcs,
                               args.train_size_per_seq, args.max_seq_per_input)
    test_data, test_cases = gen_traces(seqs[:args.n_only], funcs,
                               args.test_size_per_seq, args.max_seq_per_input)

    print(len(train_data), len(test_data))
    random.shuffle(train_data)
    random.shuffle(test_data)
    random.shuffle(test_cases)

    shutil.rmtree('numpy_model/data', True)

    model = IndependentOperatorsModel(work_dir='numpy_model')
    train_datasets = model.create_operator_datasets(train_data, mode='train')
    valid_datasets = model.create_operator_datasets(test_data, mode='valid')

    pickle.dump({k: v.path for k, v in train_datasets.items()}, open('train_datasets_pathmap.pkl', 'wb'))
    pickle.dump({k: v.path for k, v in valid_datasets.items()}, open('valid_datasets_pathmap.pkl', 'wb'))
    pickle.dump(test_cases, open('test_cases.pkl', 'wb'))


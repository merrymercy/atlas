import itertools, traceback, warnings, pickle, random, argparse
import numpy as np
from tqdm import tqdm

import atlas
from atlas.synthesis.numpy import NumpyChecker, NumpyGraphEncoder, NumpyGeneratorModel,\
        numpy_sequence_generator

from gen_data import gen_traces, gen_sequence, infinite_list, get_used_functions

def test_synthesize(inputs, output, max_n_trial, funcs, func_seq_replay):
    ct = 0

    if func_seq_replay:
        replay_map = {"func_seq": func_seq_replay}
    else:
        replay_map = {}

    for out, prog in numpy_sequence_generator.generate(inputs, output, funcs=funcs)\
                                             .with_replay(replay_map):
        ct += 1
        if NumpyChecker.check(out, output):
            return prog, ct

        if ct >= max_n_trial:
            break

    return None, ct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-only', type=int, default=1)
    parser.add_argument("--model-name", type=str, default='numpy_model')
    parser.add_argument("--mode", choices=['baseline', 'model', 'all'], default='baseline')
    args = parser.parse_args()
    
    print(args)

    seqs = gen_sequence()
    funcs = get_used_functions(seqs)

    test_cases = pickle.load(open('test_cases.pkl', 'rb'))

    if args.mode == 'baseline':
        strategies = ['dfs', 'random']
    elif args.mode == 'model':
        strategies = ['dfs-model']
    elif args.mode == 'all':
        strategies = ['dfs', 'random', 'dfs-model']

    for stra in strategies:
        if stra == 'dfs':
            numpy_sequence_generator.set_default_strategy('dfs')
        elif stra == 'random':
            numpy_sequence_generator.set_default_strategy('randomized')
        elif stra == 'dfs-model':
            model = NumpyGeneratorModel.load(args.model_name)
            numpy_sequence_generator.set_default_strategy('dfs')
            numpy_sequence_generator.set_default_model(model)
        else:
            raise ValueError(f"Invalid strategy {args.strategy}")

        success = 0
        fail = 0
        total_n_trial = 0
        for inputs, output in tqdm(test_cases):
            prog, ct = test_synthesize(inputs, output, max_n_trial=100, 
                                       funcs=funcs, 
                                       func_seq_replay=infinite_list(['meshgrid']))
            if prog is not None:
                success += 1
                total_n_trial += ct

        print(f"Strategy {stra}:  "
              f"{success}/{len(test_cases)} success, average trial {total_n_trial / max(success, 1):.1f}")


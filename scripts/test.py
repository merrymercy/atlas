import itertools, traceback, warnings, pickle, random, argparse, random

import numpy as np
from tqdm import tqdm

import atlas
from atlas.exceptions import StepTermination
from atlas.synthesis.numpy import NumpyChecker, NumpyGraphEncoder, NumpyGeneratorModel,\
        numpy_sequence_generator, NumpyRandomStrategy, NumpyDfsStrategy,\
        NumpyBeamSearchStrategy

from gen_data import gen_traces, gen_sequence, infinite_list, get_used_functions
from stackoverflow_cases import cases as stackoverflow_cases

def test_synthesize(inputs, output, stra, funcs, func_seq_replay=None):
    if func_seq_replay:
        replay_map = {"func_seq": func_seq_replay}
    else:
        replay_map = {}

    numpy_sequence_generator.set_default_strategy(stra)

    for out, prog in numpy_sequence_generator.with_env(replay=replay_map)\
                                             .generate(inputs, output, funcs=funcs):
        if NumpyChecker.check(out, output):
            return prog, stra.iter_ct + 1

    return None, stra.iter_ct

def random_filter_except_seq(sid):
    if sid == '/numpy_sequence_generator/SequenceFixed@func_seq@1':
        return False
    else:
        return True

def random_filter_all(sid):
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-only', type=int)
    parser.add_argument('--max-n-trial', type=int, default=100)
    parser.add_argument("--model-name", type=str, default='numpy_model')
    parser.add_argument("--baseline", action='store_true')
    parser.add_argument("--gnn", action='store_true')
    parser.add_argument("--gnnall", action='store_true')
    parser.add_argument("--gnnbase", action='store_true')
    parser.add_argument("--all", action='store_true')
    parser.add_argument("--single", type=str)
    parser.add_argument("--stackoverflow", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    seqs = gen_sequence()
    funcs = get_used_functions(seqs)

    test_cases = pickle.load(open('test_cases.pkl', 'rb'))
    if args.n_only:
        test_cases = test_cases[:args.n_only]

    strategies = []

    if args.all:
        strategies = ['dfs', 'random', 'dfs-model', 'random-model', 'beam-model']
    elif args.baseline:
        strategies = ['dfs', 'random']
    elif args.gnn:
        strategies = ['dfs-model']
    elif args.gnnall:
        strategies = ['dfs-model', 'dfs-model-random']
    elif args.gnnbase:
        strategies = ['dfs-model-seq-only']
    elif args.single:
        strategies = [args.single]
    else: # default
        strategies = ['dfs', 'random']

    summary = []

    if len(seqs) == 1:
        func_seq_replay = infinite_list(seqs[0][0])
    else:
        func_seq_replay = None

    if any(["model" in item for item in strategies]):
        print("Load model...")
        model = NumpyGeneratorModel()
        model.deserialize(args.model_name)
        numpy_sequence_generator.set_default_model(model)

    for item in strategies:
        if item == 'dfs':
            stra = NumpyDfsStrategy(operator_iterator_bound=10,
                                    random_filter=random_filter_all,
                                    max_iter=args.max_n_trial)
        elif item == 'random':
            stra = NumpyRandomStrategy(max_iter=args.max_n_trial)
        elif item == 'dfs-model':
            stra = NumpyDfsStrategy(operator_iterator_bound=10,
                                    max_iter=args.max_n_trial)
        elif item == 'random-model':
            stra = NumpyRandomStrategy(model=model,
                                      max_iter=args.max_n_trial)
        elif item == 'dfs-model-seq-only':
            stra = NumpyDfsStrategy(operator_iterator_bound=10,
                                    random_filter=random_filter,
                                    max_iter=args.max_n_trial)
        elif item == "beam-model":
            stra = NumpyBeamSearchStrategy(model=model,
                                           max_iter=args.max_n_trial,
                                           operator_iterator_bound=10,
                                           beam_size=args.max_n_trial * 2)
        else:
            raise ValueError(f"Invalid strategy {item}")

        success = 0
        fail = 0
        total_n_trial = 0

        if args.stackoverflow:
            pick_no = set([3, 4, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 21, 22, 25, 26, 29,
                           32, 33, 34, 35, 36, 38, 39])
            visited = set()
            test_cases = []
            trial_cts = []
            for x in stackoverflow_cases:
                if x.no in pick_no and x.no not in visited:
                    test_cases.append(([inp for inp in x.inputs], x.output))
                    visited.add(x.no)

            for inputs, output in tqdm(test_cases, ascii=True):
                try:
                    prog, ct = test_synthesize(inputs, output, stra=stra,
                                               funcs=funcs, func_seq_replay=func_seq_replay)
                    if prog is not None:
                        success += 1
                        total_n_trial += ct
                        trial_cts.append(ct)
                    else:
                        trial_cts.append(None)
                except StepTermination as e:
                    print(e.choices[:10])
                    print(e.probs[:10])
                    print("======================")

            to_print = f"Strategy {item}: "\
                       f"{success}/{len(test_cases)} success, average trial "\
                       f"{total_n_trial / max(success, 1):.1f}"\
                       f"\n {trial_cts}"
        else:
            for inputs, output in tqdm(test_cases, ascii=True):
                prog, ct = test_synthesize(inputs, output, stra=stra,
                                           funcs=funcs, func_seq_replay=func_seq_replay)
                if prog is not None:
                    success += 1
                    total_n_trial += ct

            to_print = f"Strategy {item}: "\
                       f"{success}/{len(test_cases)} success, average trial "\
                       f"{total_n_trial / max(success, 1):.1f}"

        print(to_print)
        summary.append(to_print)

    print("====================================================================")
    print("\n".join(summary))


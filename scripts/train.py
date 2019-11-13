import pickle
import argparse

import numpy as np

import atlas
from atlas.synthesis.numpy import NumpyGeneratorModel
from atlas.models.tensorflow.graphs.earlystoppers import SimpleEarlyStopper

from gen_data import gen_sequence, get_used_functions

def train_seq_only(sid):
    if sid == '/numpy_sequence_generator/SequenceFixed@func_seq@1':
        return False
    return True

def skip_seq(sid):
    return not train_seq_only(sid)

def keep_name_only(name):
    def func(sid):
        return name not in sid
    return func

def load_data():
    train_datasets_pathmap = pickle.load(open('train_datasets_pathmap.pkl', 'rb'))
    valid_datasets_pathmap = pickle.load(open('valid_datasets_pathmap.pkl', 'rb'))

    return train_datasets_pathmap, valid_datasets_pathmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='numpy_model')
    parser.add_argument("--seq-only", action='store_true')
    parser.add_argument("--skip-seq", action='store_true')
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--n-epoch", type=int, default=-1)
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--patient", action='store_true')
    args = parser.parse_args()
    print(args)

    train_datasets_pathmap, valid_datasets_pathmap = load_data()

    if args.seq_only:
        skip_sid = train_seq_only
    elif args.skip_seq:
        skip_sid = skip_seq
    elif args.name_filter is not None:
        skip_sid = keep_name_only(args.name_filter)
    else:
        skip_sid = None

    model_configs = {
        'Select':        {'batch_size': 30000, 'learning_rate': 0.001},
        'SelectFixed':   {'batch_size': 20000, 'learning_rate': 0.001},
        'Sequence':      {'batch_size': 25000, 'learning_rate': 0.001},
        'SequenceFixed': {'batch_size': 30000, 'learning_rate': 0.001},
        'OrderedSubset': {'batch_size': 25000, 'learning_rate': 0.001},
        'Subset':        {'batch_size': 25000, 'learning_rate': 0.001},
    }

    if args.patient:
        early_stopper = SimpleEarlyStopper(patience=20, 
                                           val_loss_threshold=0.02,
                                           val_acc_threshold=0.02,
                                           patience_zero_threshold=0.99)
    else:
        early_stopper = SimpleEarlyStopper(patience=20,
                                           val_loss_threshold=0.02,
                                           val_acc_threshold=0.02,
                                           patience_zero_threshold=0.95)

    if args.load:
        print("Load pre-trained models")
        model = NumpyGeneratorModel.load(args.model_name, model_configs)
    else:
        model = NumpyGeneratorModel(args.model_name, model_configs)

    train_datasets = model.load_operator_datasets(train_datasets_pathmap)
    valid_datasets = model.load_operator_datasets(valid_datasets_pathmap)

    model.train_with_datasets(train_datasets, valid_datasets, 
                              num_epochs=args.n_epoch, 
                              early_stopper=early_stopper,
                              skip_sid=skip_sid)
    model.save(args.model_name)


'''
    from atlas.synthesis.numpy.encoders import NumpyGraphEncoder
    from atlas.synthesis.utils import dump_encodings
    seqs = gen_sequence()
    funcs = get_used_functions(seqs)

    for x, dataset in train_datasets.items():
        encoded_train = dump_encodings(dataset, NumpyGraphEncoder().get_encoder(x))
        max_n_nodes, max_n_edges = 0, 0
        average_n_nodes, average_n_edges = None, None
        for i, (g, trace) in enumerate(zip(encoded_train, dataset)):
            updated = False
            if len(g['nodes']) > max_n_nodes:
                max_n_nodes = len(g['nodes'])
                updated = True
            if len(g['edges']) > max_n_edges:
                max_n_edges = len(g['edges'])
                updated = True

            if average_n_nodes is None:
                average_n_nodes = len(g['nodes'])
            else:
                average_n_nodes = 0.95 * average_n_nodes + 0.05 * len(g['nodes'])

            if average_n_edges is None:
                average_n_edges = len(g['edges'])
            else:
                average_n_edges = 0.95 * average_n_edges + 0.05 * len(g['edges'])

            if updated:
                print("Max updated", max_n_nodes, max_n_edges,
                      [funcs[idx] for idx in g['choice'][:-1]])

            n_n = len(g['nodes'])
            n_e = len(g['edges'])
            if n_e > n_n * n_n / 2 / 2:
                print("!Too Dense", n_n, n_e,
                      [funcs[idx] for idx in g['choice'][:-1]])

#            if i % 1000 == 0:
#                print("Average", average_n_nodes, average_n_edges)
        exit()
'''


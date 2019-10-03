import pickle
import argparse

import numpy as np

import atlas
from atlas.synthesis.numpy import NumpyGeneratorModel


def load_data():
    train = pickle.load(open('train_traces.pkl', 'rb'))
    test = pickle.load(open('test_traces.pkl', 'rb'))

    return train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='numpy_model')
    args = parser.parse_args()

    train_data, test_data = load_data()

    model = NumpyGeneratorModel(model_configs={
        'Select': {'batch_size': 100000, 'learning_rate': 0.005}
    })
    model.train(train_data, test_data, num_epochs=4)
    model.save(args.model_name)


import unittest
import warnings
from typing import Any, List

import numpy as np

from atlas import generator
from atlas.exceptions import ExceptionAsContinue
from atlas.utils import get_group_by_name
import atlas.synthesis.numpy
from atlas.synthesis.numpy.checker import Checker

api_gens = {
    gen.name: gen for gen in get_group_by_name('numpy')
}


@generator(group='numpy')
def simple_enumerator(inputs, output, func_seq):
    prog = []
    intermediates = []
    for func in func_seq:
        func = api_gens[func]
        try:
            val, args = func(intermediates + inputs, output)
        except Exception as e:
            warnings.warn(str(e))
            raise ExceptionAsContinue
        prog.append((func.name, args))
        intermediates.append(val)

    return intermediates[-1], prog


class TestGenerators(unittest.TestCase):
    def check(self, inputs: List[Any], output: Any, funcs: List[str], seqs: List[List[int]],
              constants: List[Any] = None):
        if constants is not None:
            inputs += constants

        func_seqs = [[funcs[i] for i in seq] for seq in seqs]
        for func_seq in func_seqs:
            for val, prog in simple_enumerator.generate(inputs, output, func_seq):
                if Checker.check(output, val):
                    return True

        self.assertTrue(False, "Did not find a solution")

    # Array Creation Routines
    def test_arange(self):
        inputs = []
        output = np.arange(2, 5)
        self.check(inputs, output, ['arange'], [[0]])

    def test_eye(self):
        inputs = []
        output = np.eye(3, 4)
        self.check(inputs, output, ['eye'], [[0]])

    def test_identity(self):
        inputs = []
        output = np.identity(5)
        self.check(inputs, output, ['identity'], [[0]])

    def test_ones(self):
        inputs = []
        output = np.ones((2, 2))
        self.check(inputs, output, ['ones'], [[0]])

    def test_ones_like(self):
        inputs = [np.random.randn(2, 5)]
        output = np.ones_like(inputs[0])
        self.check(inputs, output, ["ones_like"], [[0]])

    def test_full(self):
        inputs = []
        output = np.full((2, 2), 2.0)
        self.check(inputs, output, ['full'], [[0]])

    def test_full_like(self):
        inputs = [np.random.randn(2, 5)]
        output = np.full_like(inputs[0], 3.1)
        self.check(inputs, output, ["full_like"], [[0]])

    def test_reshape(self):
        inputs = [np.random.randn(3, 10)]
        output = np.reshape(inputs[0], [5, 2, 3])
        self.check(inputs, output, ['reshape'], [[0]])

    def test_ravel(self):
        inputs = [np.random.randn(2, 4)]
        output = np.ravel(inputs[0], order='A')
        self.check(inputs, output, ['ravel'], [[0]])

    def test_flatten(self):
        inputs = [np.random.randn(3, 4)]
        output = inputs[0].flatten()
        self.check(inputs, output, ['ndarray.flatten'], [[0]])

    def test_moveaxis(self):
        inputs = [np.random.randn(3, 4, 5)]
        output = np.moveaxis(inputs[0], [1], [2])
        self.check(inputs, output, ['moveaxis'], [[0]])

    def test_rollaxis(self):
        inputs = [np.random.randn(3, 4, 5)]
        output = np.rollaxis(inputs[0], 1, 1)
        self.check(inputs, output, ['rollaxis'], [[0]])

    def test_swapaxes(self):
        inputs = [np.random.randn(3, 4, 5)]
        output = np.swapaxes(inputs[0], 0, 2)
        self.check(inputs, output, ['swapaxes'], [[0]])

    def test_transpose(self):
        inputs = [np.random.randn(3, 4, 5)]
        output = np.transpose(inputs[0], axes=[0, 2, 1])
        self.check(inputs, output, ['transpose'], [[0]])

    def test_expand_dims(self):
        inputs = [np.random.randn(3, 4, 5)]
        output = np.expand_dims(inputs[0], axis=2)
        self.check(inputs, output, ['expand_dims'], [[0]])

    def test_squeeze(self):
        inputs = [np.random.randn(3, 1, 5)]
        output = np.squeeze(inputs[0])
        self.check(inputs, output, ['squeeze'], [[0]])

    def test_concatenate(self):
        inputs = [np.random.randn(2, 3), np.random.randn(1, 3)]
        output = np.concatenate(inputs, axis=0)
        self.check(inputs, output, ['concatenate'], [[0]])

    def test_stack(self):
        inputs = [np.random.randn(2, 3), np.random.rand(2, 3)]
        output = np.stack(inputs, 0)
        self.check(inputs, output, ['stack'], [[0]])

    def test_column_stack(self):
        inputs = [np.random.randn(2,), np.random.rand(2,)]
        output = np.column_stack(inputs)
        self.check(inputs, output, ['column_stack'], [[0]])

    def test_dstack(self):
        inputs = [np.random.randn(2, 3), np.random.rand(2, 3)]
        output = np.dstack(inputs)
        self.check(inputs, output, ['dstack'], [[0]])

    def test_hstack(self):
        inputs = [np.random.randn(5), np.random.randn(9)]
        output = np.hstack(inputs)
        self.check(inputs, output, ['hstack'], [[0]])

    def test_vstack(self):
        inputs = [np.random.randn(5), np.random.randn(5)]
        output = np.vstack(inputs)
        self.check(inputs, output, ['vstack'], [[0]])

    def test_split(self):
        inputs = [np.random.randn(4, 4, 3)]
        output = np.split(inputs[0], 2, axis=0)
        self.check(inputs, output, ['split'], [[0]])

    def test_array_split(self):
        inputs = [np.random.randn(4, 4, 3)]
        output = np.array_split(inputs[0], 3)
        self.check(inputs, output, ['split'], [[0]])

    def test_dsplit(self):
        inputs = [np.random.randn(4, 4, 4)]
        output = np.dsplit(inputs[0], 2)
        self.check(inputs, output, ['dsplit'], [[0]])

    def test_hsplit(self):
        inputs = [np.random.randn(4, 4, 3)]
        output = np.hsplit(inputs[0], 2)
        self.check(inputs, output, ['hsplit'], [[0]])

    def test_vsplit(self):
        inputs = [np.random.randn(4, 4, 3)]
        output = np.vsplit(inputs[0], 2)
        self.check(inputs, output, ['vsplit'], [[0]])

    def test_tile(self):
        inputs = [np.random.randn(4, 3, 2)]
        output = np.tile(inputs[0], [1, 2, 1])
        self.check(inputs, output, ["tile"], [[0]])

    def test_repeat(self):
        inputs = [np.random.randn(3, 4, 2)]
        output = np.repeat(inputs[0], 3, axis=1)
        self.check(inputs, output, ["repeat"], [[0]])

    def test_ndarray_equal(self):
        inputs = [np.array([2, -1, 4])]
        output = (inputs[0] == -1)
        print(output)
        self.check(inputs, output, ["ndarray.__equal__"], [[0]])

    def test_unique(self):
        inputs = [np.array([2, 3, 2])]
        output = np.unique(inputs[0])
        self.check(inputs, output, ["unique"], [[0]])

    def test_meshgrid(self):
        inputs = [np.array([3, 2, 4]), np.array([4, 2, 1])]
        output = np.meshgrid(inputs[0], inputs[1])
        self.check(inputs, output, ['meshgrid'], [[0]])

    def test_linalg_norm(self):
        inputs = [np.array([3, 4, 5,2])]
        output = np.linalg.norm(inputs[0])
        self.check(inputs, output, ['linalg.norm'], [[0]])

    def test_ndarray_sub(self):
        inputs = [np.array([3, 2, 1]), np.array([2, 4, 9])]
        output = inputs[1] - inputs[0]
        self.check(inputs, output, ['ndarray.__sub__'], [[0]])

    def test_matmul(self):
        inputs = [np.array([[3, 2], [2, 4]]), np.array([[2, 9], [4, 2]])]
        output = np.matmul(inputs[1], inputs[0])
        self.check(inputs, output, ['matmul'], [[0]])

    def test_sum(self):
        inputs = [np.array([[2,3,4], [3,2,1]])]
        output = np.sum(inputs[0])
        self.check(inputs, output, ['sum'], [[0]])

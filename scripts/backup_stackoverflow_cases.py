from collections import namedtuple
import numpy as np

from atlas.synthesis.numpy import NumpyChecker

Case = namedtuple("Case", ['no', 'inputs', 'output', 'sol_func', 'func_sequence'])

cases = []

def register_case(no, inputs, output, func_sequence):
    def wrapper(func):
        cases.append(Case(no, inputs, output, func, func_sequence)) 

    return wrapper

@register_case(no=2,
    inputs=[np.array([[1, -1], [2, -1], [2, -2], [3, -3]])],
    output=np.array([[1, 5], [2, 5], [2, 10], [3, 10]]),
    func_sequence=['split', 'ndarray.__equal__', 'where', 'hstack'])
def func(inputs):
    a = inputs[0]
    b, c = np.split(a, 2, axis=1)
    d = (c == -1)             # where
    e = np.where(d, 5, 10)
    f = np.hstack((b, e))
    return f

@register_case(no=3,
    inputs=[np.array(
        [[1, 1, 1, 0, 0, 0],
         [0, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 0]])],
    output=np.array(
        [[0, 1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 0]]),
    func_sequence=['unique'])
def func(inputs):
    a = inputs[0]
    b = np.unique(a, axis=0)
    return b

@register_case(no=4,
    inputs=[np.array([1, 2, 3]), np.array([4, 5])],
    output=np.array([[1,4], [2,4], [3,4], [1,5], [2,5], [3,5]]),
    func_sequence=['tile', 'repeat', 'vstack', 'transpose'])
def func(inputs):
    a, b = inputs
    c = np.tile(a, len(b))
    d = np.repeat(b, len(a))
    e = np.vstack([c, d])
    f = np.transpose(e)
    return f

@register_case(no=4,
    inputs=[np.array([1, 2, 3]), np.array([4, 5])],
    output=np.array([[1,4], [2,4], [3,4], [1,5], [2,5], [3,5]]),
    func_sequence=['meshgrid', 'dstack', 'reshape'])
def func(inputs):
    a, b = inputs
    c, d = np.meshgrid(a, b)
    e = np.dstack((c, d))
    f = np.reshape(e, (6, 2))
    return f

@register_case(no=6,
    inputs=[np.array([2, 3, 4]), np.array([4, 5, 5])],
    output=3.0,
    func_sequence=['ndarray.__sub__', 'linalg.norm'])
def func(inputs):
    a, b = inputs
    c = a - b
    d = np.linalg.norm(c)
    return d


@register_case(no=8,
    inputs=[np.array([[5, 1 ,3], [1, 1 ,1], [1, 2 ,1]]), np.array([1, 2, 3])],
    output=np.array([16, 6, 8]),
    func_sequence=['matmul'])
def func(inputs):
    a, b = inputs
    c = np.matmul(a, b)
    return c

@register_case(no=9,
    inputs=[np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])],
    output=np.array([18, 22, 26]),
    func_sequence=['sum'])
def func(inputs):
    a = inputs[0]
    b = np.sum(a, axis=0)
    return b

@register_case(no=10,
    inputs=[np.array([1400, 1500, 1600, np.nan, np.nan, np.nan, 1700])],
    output=np.array([1400.0, 1500.0, 1600.0, 1700.0]),
    func_sequence=['isnan', 'logical_not', 'compress'])
def func(inputs):
    a = inputs[0]
    b = np.isnan(a)
    c = np.logical_not(b)
    d = np.compress(c, a)
    return d

@register_case(no=12,
    inputs=[np.array([4, 5, 6, 8])],
    output=np.array([0.33686077, 0.42107596, 0.50529115, 0.67372154]),
    func_sequence=['linalg.norm', 'ndarray.__div__'])
def func(inputs):
    a = inputs[0]
    b = np.linalg.norm(a)
    c = a / b
    return c

@register_case(no=13,
    inputs=[np.array([[1,2,3], [4,5,6]])],
    output=np.array([1, 2, 3, 4, 5, 6]),
    func_sequence=['ndarray.flatten'])
def func(inputs):
    a = inputs[0]
    b = np.ndarray.flatten(a)
    return b

@register_case(no=14,
    inputs=[np.array([1,2,3]), np.array([4,5,6])],
    output=np.array([[1, 2, 3], [4, 5, 6]]),
    func_sequence=['vstack'])
def func(inputs):
    a, b = inputs
    c = np.vstack((a, b))
    return c

@register_case(no=15,
    inputs=[np.array([[0, 1], [2 ,3], [4 ,5], [6 ,7], [8 ,9], [10 ,11], [12 ,13], [14 ,15], [16 ,17]])],
    output=np.array([[[0, 6, 12], [2, 8, 14], [4, 10, 16]], [[1, 7, 13], [3, 9, 15], [5, 11, 17]]]),
    func_sequence=['reshape', 'swapaxes'])
def func(inputs):
    a = inputs[0]
    b = np.reshape(a, (3, 3, 2))
    c = np.swapaxes(b, 0, 2)
    return c

@register_case(no=16,
    inputs=[np.array([[10,50,30],[60,20,40]])],
    output=(1, 0),
    func_sequence=['argmax', 'unravel_index'])
def func(inputs):
    a = inputs[0]
    b = np.argmax(a)
    c = np.unravel_index(a.argmax(), a.shape)
    return c

@register_case(no=17,
    inputs=[np.array([1, 3, 5]), np.array([2, 4, 6])],
    output=np.array([1, 2, 3, 4, 5, 6]),
    func_sequence=['vstack', 'reshape'])
def func(inputs):
    a, b = inputs
    c = np.vstack((a, b))
    d = np.reshape(c, (-1,), order='F')
    return d

@register_case(no=17,
    inputs=[np.array([1, 3, 5]), np.array([2, 4, 6])],
    output=np.array([1, 2, 3, 4, 5, 6]),
    func_sequence=['dstack', 'ravel'])
def func(inputs):
    a, b = inputs
    c = np.dstack((a, b))
    d = np.ravel(c)
    return d

@register_case(no=21,
    inputs=[np.array([1, 3, 2, 4, 5, 0, -1])],
    output=np.array([4, 3, 1]),
    func_sequence=['argsort', 'flip', 'arange', 'take'])
def func(inputs):
    a = inputs[0]
    b = np.argsort(a)
    c = np.flip(b)
    d = np.arange(3)
    e = np.take(c, d)
    return e

@register_case(no=22,
    inputs=[np.array([[9, 2, 3], [4, 5, 6], [7, 0, 5]])],
    output=np.array([[7, 0, 5], [9, 2, 3], [4, 5,6]]),
    func_sequence=['take', 'argsort', 'take'])
def func(inputs):
    a = inputs[0]
    b = np.take(a, 1, axis=1)
    c = np.argsort(b)
    d = np.take(a, c, axis=0)
    return d

if __name__ == "__main__":
    unique_cases = set()
    for case in cases:
        unique_cases.add(case.no)

    print(f"Load {len(unique_cases)} cases...")
    for case in cases:
        actual = case.sol_func(case.inputs)
        if not NumpyChecker.check(actual, case.output):
            print(f"\nCase {case.no} Failed: ")
            print(f"Actual: {type(actual)}\n{actual}")
            print(f"Expected: {type(case.output)}\n{case.output}")


from collections import namedtuple
import numpy as np

from atlas.synthesis.numpy import NumpyChecker

Case = namedtuple("Case", ['no', 'inputs', 'output', 'sol_func', 'func_sequence'])

cases = []

def register_case(no, inputs, output, func_sequence):
    def wrapper(func):
        case = Case(no, inputs, output, func, func_sequence)
        if case.output is None:
            out = case.sol_func(inputs)
            case = Case(no, inputs, out, func, func_sequence)
        cases.append(case)

    return wrapper

#@register_case(no=2,
#    inputs=[np.array([[1, -1], [2, -1], [2, -2], [3, -3]])],
#    output=np.array([[1, 5], [2, 5], [2, 10], [3, 10]]),
#    func_sequence=['split', 'ndarray.__equal__', 'where', 'hstack'])
#def func(inputs):
#    a = inputs[0]
#    b, c = np.split(a, 2, axis=1)
#    d = (c == -1)             # where
#    e = np.where(d, 5, 10)
#    f = np.hstack((b, e))
#    return f

@register_case(no=3,
    inputs=[np.array(
        [[1.1, 1.1, 1.1, 0.1, 0.1, 0.1],
         [0.1, 1.1, 1.1, 1.1, 0.1, 0.1],
         [0.1, 1.1, 1.1, 1.1, 0.1, 0.1],
         [1.1, 1.1, 1.1, 0.1, 0.1, 0.1],
         [1.1, 1.1, 1.1, 1.1, 1.1, 0.1]])],
    output=np.array(
        [[0.1, 1.1, 1.1, 1.1, 0.1, 0.1],
         [1.1, 1.1, 1.1, 0.1, 0.1, 0.1],
         [1.1, 1.1, 1.1, 1.1, 1.1, 0.1]]),
    func_sequence=['unique'])
def func(inputs):
    a = inputs[0]
    b = np.unique(a, axis=0)
    return b

@register_case(no=4,
    inputs=[np.array([1.1, 2.1, 3.1]), np.array([4.1, 5.1])],
    output=np.array([[1.1,4.1], [2.1,4.1], [3.1,4.1], [1.1,5.1], [2.1,5.1], [3.1,5.1]]),
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
    inputs=[np.array([2.1, 3.1, 4.1]), np.array([4.1, 5.1, 5.1])],
    output=None, #3.0,
    func_sequence=['ndarray.__sub__', 'linalg.norm'])
def func(inputs):
    a, b = inputs
    c = a - b
    d = np.linalg.norm(c)
    return d


@register_case(no=8,
    #inputs=[np.array([[5.1, 1.1 ,3.1], [1.1, 1.1 ,1.1], [1.1, 2.1 ,1.1]]), np.array([1.1, 2.1, 3.1])],
    inputs=[np.array([[5.1, 3.1 ,2.1], [1.1, 9.1 ,8.1], [10.1, 7.1 ,3.1]]), np.array([1.2, 2.2, 3.2])],
    output=None, # np.array([16, 6, 8]),
    func_sequence=['matmul'])
def func(inputs):
    a, b = inputs
    c = np.matmul(a, b)
    return c

@register_case(no=9,
    inputs=[np.array([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1], [6.1, 7.1, 8.1], [9.1, 10.1, 11.1]])],
    output=None,
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
    inputs=[np.array([4.1, 5.1, 6.1, 8.1])],
    output=None,
    func_sequence=['linalg.norm', 'ndarray.__div__'])
def func(inputs):
    a = inputs[0]
    b = np.linalg.norm(a)
    c = a / b
    return c

@register_case(no=13,
    inputs=[np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])],
    output=None,
    func_sequence=['ndarray.flatten'])
def func(inputs):
    a = inputs[0]
    b = np.ndarray.flatten(a)
    return b

@register_case(no=14,
    inputs=[np.array([1.1,2.1,3.1]), np.array([4.1,5.1,6.1])],
    output=None,
    func_sequence=['vstack'])
def func(inputs):
    a, b = inputs
    c = np.vstack((a, b))
    return c

@register_case(no=15,
    inputs=[np.array([[0.1, 1.1], [2.1, 3.1], [4.1, 5.1], [6.1, 7.1], [8.1, 9.1], [10.1, 11.1], [12.1, 13], [14.1, 15], [16.1, 17]])],
    output=None,
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
    c = np.unravel_index(b, a.shape)
    return c

@register_case(no=17,
    inputs=[np.array([1.1, 3.1, 5.1]), np.array([2.1, 4.1, 6.1])],
    output=None,
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


@register_case(no=21,
    inputs=[np.array([1, 3, 2, 4, 5, 0, -1])],
    output=np.array([4, 3, 1]),
    func_sequence=['argsort', 'flip', 'ndarray.__getitem__'])
def func(inputs):
    a = inputs[0]
    b = np.argsort(a)
    c = np.flip(b)
    d = c[0:3]
    return d


@register_case(no=22,
    inputs=[np.array([[9, 2, 3], [4, 5, 6], [7, 0, 5]])],
    output=np.array([[7, 0, 5], [9, 2, 3], [4, 5, 6]]),
    func_sequence=['take', 'argsort', 'take'])
def func(inputs):
    a = inputs[0]
    b = np.take(a, 1, axis=1)
    c = np.argsort(b)
    d = np.take(a, c, axis=0)
    return d


@register_case(no=25,
    inputs=[np.array([[1, 2, 3, 4], [5, 6, 7, 8]])],
    output=np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]]]),
    func_sequence=['reshape', 'swapaxes', 'reshape'])
def func(inputs):
    a = inputs[0]
    b = np.reshape(a, (2, 2, 1, 2))
    c = np.swapaxes(b, 0, 1)
    d = np.reshape(c, (2, 2, 2))
    return d


@register_case(no=26,
    inputs=[np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])],
    output=np.array([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0]]),
    func_sequence=['zeros', 'hstack'])
def func(inputs):
    a = inputs[0]
    b = np.zeros((2, 1))
    c = np.hstack((a, b))
    return c


@register_case(no=29,
    inputs=[np.array([1,1,1,2,2,2,5,25,1,1])],
    output=np.array([[1, 5], [2,3], [5,1], [25,1]]),
    func_sequence=['unique', 'vstack', 'transpose'])
def func(inputs):
    a = inputs[0]
    b, c = np.unique(a, return_counts=True)
    d = np.vstack((b, c))
    e = np.transpose(d)
    return e

@register_case(no=32,
    inputs=[np.array([[1], [2], [3], [4]])],
    output=np.array([1, 2, 3, 4]),
    func_sequence=['ndarray.flatten'])
def func(inputs):
    a = inputs[0]
    b = a.flatten()
    return b

@register_case(no=33,
    inputs=[np.array([1, 2, 3]), np.array([4, 5, 6])],
    output=np.array([[1, 2, 3], [4, 5, 6]]),
    func_sequence=['vstack'])
def func(inputs):
    a = inputs[0]
    b = inputs[1]
    c = np.vstack((a, b))
    return c

@register_case(no=34,
    inputs=[np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([-1])],
    output=np.array([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]]),
    func_sequence=['repeat', 'reshape', 'hstack'])
def func(inputs):
    a = inputs[0]
    b = inputs[1]
    c = np.repeat(b, 3)
    d = np.reshape(c, (3, 1))
    e = np.hstack((a, d))
    return e

@register_case(no=35,
    inputs=[np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([2, 3, 6])],
    output=np.array([1, 2, 5, 6, 8, 9]),
    func_sequence=['delete'])
def func(inputs):
    a = inputs[0]
    b = inputs[1]
    c = np.delete(a, b)
    return c

@register_case(no=36,
    inputs=[np.array([[1, 2, 3], [4, 3, 1]])],
    output=np.array([1, 1, 0]),
    func_sequence=['argmax'])
def func(inputs):
    a = inputs[0]
    b = np.argmax(a, axis=0)
    return b

@register_case(no=38,
    inputs=[np.array([1,2,3,4,5,6,7,8,9]), np.array([1, 7, 10])],
    output=np.array([0, 6, 9]),
    func_sequence=['searchsorted'])
def func(inputs):
    a = inputs[0]
    b = inputs[1]
    c = np.searchsorted(a, b)
    return c

@register_case(no=39,
    inputs=[np.array([12, 989, 13, 1, 997]), np.array([255])],
    output=np.array([12, 255, 13, 1, 255]),
    func_sequence=['minimum'])
def func(inputs):
    a = inputs[0]
    b = inputs[1]
    c = np.minimum(a, b)
    return c

if __name__ == "__main__":
    unique_cases = set()
    for case in cases:
        unique_cases.add(case.no)

    print(f"Load {len(unique_cases)} cases...")
    print(unique_cases)
    for case in cases:
        actual = case.sol_func(case.inputs)
        if not NumpyChecker.check(actual, case.output):
            print(f"\nCase {case.no} Failed: ")
            print(f"Actual: {type(actual)}\n{actual}")
            print(f"Expected: {type(case.output)}\n{case.output}")

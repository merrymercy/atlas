import logging
from typing import Callable

import pandas as pd
import numpy as np

from ...generators import generator
from ...stubs import Select, Sequence, Subset, OrderedSubset, Product
from ..utils import wrap_io_context
from .utils import get_non_1_prime_factors

## Configs for random ndarray
MAX_N_DIM = 10        # Maximum number of dimensions
MIN_N_DIM = 1         # Minimum number of dimensions
MIN_DIM_LENGTH = 1    # Minimum length per dimension
MAX_DIM_LENGTH = 10   # Maximum length per dimension
CAND_DTYPES = set([np.dtype('float')])  #, np.dtype('int')])

# Configs for some APIs
MAX_REPEATS = 8       # max value for arg `repeats` in numpy.repeat
MAX_REPS_N_DIM = 5    # max number of dimensions in numpy.tile
MAX_REPS_PER_DIM = 5  # max tile number for each dimensions in numpy.tile

# Array Creation Routines
@generator(group='numpy', name='arange')
def gen_arange(inputs, output, **kwargs):
    c = {"O": output}

    _start = Select(range(MAX_DIM_LENGTH), context=c)
    _stop = Select(range(_start + 1, MAX_DIM_LENGTH + 1), context=c)
    _step = Select(range(1, _stop - _start + 1), context=c) # incorporate more constraints here?
    _dtype = Select(list(CAND_DTYPES.union([output.dtype] if isinstance(output, np.ndarray) else [])), context=c)

    prog = {"start": _start, "stop": _stop, "step": _step, "dtype": _dtype}
    return np.arange(**prog), prog


@generator(group='numpy', name='eye')
def gen_eye(inputs, output, **kwargs):
    c = {"O": output}

    _N = Select(range(1, MAX_DIM_LENGTH), context=c)
    _M = Select([None] + list(range(1, MAX_DIM_LENGTH)), context=c)
    _k = Select(range(-MAX_DIM_LENGTH, MAX_DIM_LENGTH+1), context=c)
    _dtype = Select(list(CAND_DTYPES.union([output.dtype] if isinstance(output, np.ndarray) else [])),
                    context=c)
    _order = Select(["C", "F"], context=c)

    return np.eye(_N, M=_M, k=_k, dtype=_dtype, order=_order), {
        "N": _N, "M": _M, "k": _k, "dtype": _dtype, "order": _order
    }


@generator(group='numpy', name='identity')
def gen_identity(inputs, output, **kwargs):
    c = {"O": output}

    _n = Select(range(MAX_DIM_LENGTH), context=c)
    _dtype = Select(list(CAND_DTYPES.union([output.dtype] if isinstance(output, np.ndarray) else [])),
                    context=c)

    return np.identity(_n, dtype=_dtype), {
        "n": _n, "dtype": _dtype,
    }


@generator(group='numpy', name='ones')
def gen_ones(inputs, output, **kwargs):
    c = {"O": output}

    n_dim = Select(range(1, MAX_N_DIM), context=c)
    _shape = []
    for i in range(n_dim):
        _shape.append(Select(range(1, MAX_DIM_LENGTH), context=c))

    _dtype = Select(list(CAND_DTYPES.union([output.dtype] if isinstance(output, np.ndarray) else [])),
                    context=c)
    _order = Select(["C", "F"], context=c)

    return np.ones(_shape, dtype=_dtype, order=_order), {
        "shape": _shape, "dtype": _dtype, "order": _order
    }


@generator(group='numpy', name='ones_like')
def gen_ones_like(inputs, output, **kwargs):
    c = {"O": output}

    _a = Select(inputs, context=c)
    _dtype = Select(list(CAND_DTYPES.union([output.dtype] if isinstance(output, np.ndarray) else [])),
                    context=c)
    _order = Select(["C", "F"], context=c)

    return np.ones_like(_a, dtype=_dtype, order=_order), {
        "a": _a, "dtype": _dtype, "order": _order
    }


@generator(group='numpy', name='full')
def gen_full(inputs, output, **kwargs):
    c = {"O": output}

    n_dim = Select(range(1, MAX_N_DIM), context=c)
    _shape = []
    for i in range(n_dim):
        _shape.append(Select(range(1, MAX_DIM_LENGTH), context=c))

    # todo(lmzheng): How to pick the value
    if output is None:
        _fill_value = Select(range(10))
    else:
        _fill_value = Select([np.min(output), np.max(output), np.mean(output), np.random.choice(output.flatten())],
                             context=c)

    _dtype = Select(list(CAND_DTYPES.union([output.dtype] if isinstance(output, np.ndarray) else [])),
                    context=c)
    _order = Select(["C", "F"], context=c)

    return np.full(_shape, _fill_value, dtype=_dtype, order=_order), {
        "shape": _shape, "fill_value": _fill_value, "dtype": _dtype, "order": _order
    }


@generator(group='numpy', name='full_like')
def gen_full_like(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O":output}

    if output is None:
        _fill_value = Select(range(10))
    else:
        _fill_value = Select([np.min(output), np.max(output), np.mean(output), np.random.choice(output.flatten())],
                             context=c)

    _dtype = Select(list(CAND_DTYPES.union([output.dtype] if isinstance(output, np.ndarray) else [])),
                    context=c)
    _subok = Select([True, False])
    _order = Select(["C", "F"], context=c)

    return np.full_like(_a, _fill_value, dtype=_dtype, order=_order), {
        "a": _a, "fill_value": _fill_value, "dtype": _dtype, "order": _order
    }


@generator(group='numpy', name='reshape')
def gen_reshape(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}

    # How many dimensions does one want to reshape to?
    n_dims = Select(range(MIN_N_DIM, MAX_N_DIM), context=c)
    if n_dims == 1:
        _newshape = _a.size
    else:
        non_1_factors = get_non_1_prime_factors(_a.size)

        _newshape = []
        for i in range(n_dims - 1):
            factors_to_use = Subset(list(non_1_factors), include_empty=True, context=c)
            dimension_size = 1
            for f in factors_to_use:
                dimension_size *= f
                non_1_factors.remove(f)
            _newshape.append(dimension_size)
        # Make sure in the end we get all the factors
        dimension_size = 1
        for f in non_1_factors:
            dimension_size *= f
        _newshape.append(dimension_size)

    _order = Select(['C', 'F', 'A'], context=c)

    return np.reshape(_a, _newshape, order=_order), {
        'a': _a, 'newshape': _newshape, 'order': _order
    }


@generator(group='numpy', name='ravel')
def gen_ravel(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}
    _order = Select(['C', 'F', 'A'], context=c)

    return np.ravel(_a, order=_order), {
        "a": _a, "order": _order
    }


@generator(group='numpy', name='ndarray.flatten')
def gen_flatten(inputs, output, **kwargs):
    _self = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _self, "O": output}
    _order = Select(["C", "F", "A", "K"], context=c)

    return np.ndarray.flatten(_self, order=_order), {
        "self": _self, "order": _order
    }


@generator(group='numpy', name='moveaxis')
def gen_moveaxis(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}

    _source = OrderedSubset(range(_a.ndim), context=c)
    _destination = OrderedSubset(_source, lengths=[len(_source)], context=c)

    return np.moveaxis(_a, _source, _destination), {
        "a": _a, "source": _source, "destination": _destination
    }


@generator(group='numpy', name='rollaxis')
def gen_rollaxis(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}
    _axis = Select(range(_a.ndim), context=c)
    _start = Select(range(_a.ndim), context=c)

    return np.rollaxis(_a, _axis, start=_start), {
        "a": _a, "axis": _axis, "start": _start
    }


@generator(group='numpy', name='swapaxes')
def gen_swapaxes(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}
    _axis1 = Select(range(_a.ndim), context=c)
    _axis2 = Select(range(_a.ndim), context=c)

    return np.swapaxes(_a, _axis1, _axis2), {
        "a": _a, "axis1": _axis1, "axis2": _axis2
    }


@generator(group='numpy', name='transpose')
def gen_transpose(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}

    _axes = None
    use_axis = Select([True, False])
    if use_axis:
        _axes = OrderedSubset(range(_a.ndim), lengths=[_a.ndim])

    return np.transpose(_a, axes=_axes), {
        "a": _a, "axes": _axes
    }


@generator(group='numpy', name='expand_dims')
def gen_expand_dims(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}
    _axis = Select(range(_a.ndim + 1), context=c)
    return np.expand_dims(_a, _axis), {
        "a": _a, "axis": _axis
    }


@generator(group='numpy', name='squeeze')
def gen_squeeze(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}

    use_axis = Select([True, False])
    if use_axis:
        size_one_dims = []
        for i in range(_a.ndim):
            if _a.shape[i] == 1:
                size_one_dims.append(i)

        if size_one_dims:
            _axis = Subset(size_one_dims, context=c)
        else:
            _axis = None
    else:
        _axis = None

    return np.squeeze(_a, _axis), {
        "a": _a, "axis": _axis
    }


@generator(group='numpy', name='concatenate')
def gen_concatenate(inputs, output, **kwargs):
    # todo(lmzheng) x 6: How to incorporate the constraints here
    # M1: Don't do anything
    # M2: Check and fallback to axis=None if it is invalid
    # M3: Enumerate all possible combinations. Constructs the list in the generator

    _arrays = OrderedSubset([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"O": output}
    for i, v in enumerate(_arrays):
        c[f"I{i}"] = v

    use_axis = Select([True, False], context=c)
    if use_axis:
        _axis = Select(range(min([x.ndim for x in _arrays])), context=c)
    else:
        _axis = None

    _out = None

    return np.concatenate(_arrays, axis=_axis, out=_out), {
        "arrays": _arrays, "axis": _axis, "out": _out
    }


@generator(group='numpy', name='stack')
def gen_stack(inputs, output, **kwargs):
    _arrays = OrderedSubset([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"O": output}
    for i, v in enumerate(_arrays):
        c[f"I{i}"] = v

    _axis = Select(range(_arrays[0].ndim + 1), context=c)

    _out = None
    return np.stack(_arrays, _axis, out=_out), {
        "arrays": _arrays, "axis": _axis, "out": _out
    }


@generator(group='numpy', name='column_stack')
def gen_column_stack(inputs, output, **kwargs):
    ct = 0
    c = {"O": output}

    domain = []
    for i, v in enumerate(inputs):
        if isinstance(v, np.ndarray):
            c[f"I{ct}"] = inputs[i]
            ct += 1
            domain.append(v)

    _tup = OrderedSubset(domain, context=c)

    return np.column_stack(_tup), {
        "tup": _tup,
    }


@generator(group='numpy', name='dstack')
def gen_dstack(inputs, output, **kwargs):
    ct = 0
    c = {"O": output}
    domain = []
    for i, v in enumerate(inputs):
        if isinstance(v, np.ndarray):
            c[f"I{ct}"] = inputs[i]
            ct += 1
            domain.append(v)

    _tup = OrderedSubset(domain, context=c)

    return np.dstack(_tup), {
        "tup": _tup
    }


@generator(group='numpy', name='hstack')
def gen_hstack(inputs, output, **kwargs):
    ct = 0
    c = {"O": output}
    domain = []
    for i, v in enumerate(inputs):
        if isinstance(v, np.ndarray):
            c[f"I{ct}"] = inputs[i]
            ct += 1
            domain.append(v)

    _tup = OrderedSubset(domain, context=c)

    return np.hstack(_tup), {
        "tup": _tup
    }


@generator(group='numpy', name='vstack')
def gen_vstack(inputs, output, **kwargs):
    ct = 0
    c = {"O": output}
    domain = []
    for i, v in enumerate(inputs):
        if isinstance(v, np.ndarray):
            c[f"I{ct}"] = inputs[i]
            ct += 1
            domain.append(v)

    _tup = OrderedSubset(domain, context=c)

    return np.vstack(_tup), {
        "tup": _tup
    }


@generator(group='numpy', name='split')
def gen_split(inputs, output, **kwargs):
    _ary = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _ary, "O": output}
    use_sections = Select([True, False], context=c)
    _axis = Select(range(_ary.ndim))

    if use_sections:
        _indices_or_sections = Select(set(get_non_1_prime_factors(_ary.shape[_axis])), context=c)
    else:
        _indices_or_sections = Subset(range(_ary.shape[_axis]), context=c)
        # todo(lmzheng): the set is too large?

    return np.split(_ary, _indices_or_sections, axis=_axis), {
        "ary": _ary, "indices_or_sections": _indices_or_sections, "axis": _axis
    }


@generator(group='numpy', name='array_split')
def gen_array_split(inputs, output, **kwargs):
    _ary = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _ary, "O": output}
    use_sections = Select([True, False], context=c)
    _axis = Select(range(_ary.ndim))

    if use_sections:
        _indices_or_sections = Select(range(2, _ary.shape[_axis]))
    else:
        _indices_or_sections = Subset(range(_ary.shape[_axis]))

    return np.array_split(_ary, _indices_or_sections, axis=_axis), {
        "ary": _ary, "indices_or_sections": _indices_or_sections, "axis": _axis
    }


@generator(group='numpy', name='dsplit')
def gen_dsplit(inputs, output, **kwargs):
    _ary = Select([inp for inp in inputs if isinstance(inp, np.ndarray) and inp.ndim >= 3])

    c = {"I0": _ary, "O": output}
    use_sections = Select([True, False], context=c)
    _axis = 2

    factors = set(get_non_1_prime_factors(_ary.shape[_axis]))
    if use_sections and factors:
        _indices_or_sections = Select(factors, context=c)
    else:
        _indices_or_sections = Subset(range(_ary.shape[_axis]), context=c)

    return np.dsplit(_ary, _indices_or_sections), {
        "ary": _ary, "indices_or_sections": _indices_or_sections
    }


@generator(group='numpy', name='hsplit')
def gen_hsplit(inputs, output, **kwargs):
    _ary = Select([inp for inp in inputs if isinstance(inp, np.ndarray) and inp.ndim >= 2])

    c = {"I0": _ary, "O": output}
    use_sections = Select([True, False], context=c)
    _axis = 1

    factors = set(get_non_1_prime_factors(_ary.shape[_axis]))
    if use_sections and factors:
        _indices_or_sections = Select(factors, context=c)
    else:
        _indices_or_sections = Subset(range(_ary.shape[_axis]), context=c)

    return np.hsplit(_ary, _indices_or_sections), {
        "ary": _ary, "indices_or_sections": _indices_or_sections
    }


@generator(group='numpy', name='vsplit')
def gen_vsplit(inputs, output, **kwargs):
    _ary = Select([inp for inp in inputs if isinstance(inp, np.ndarray) and inp.ndim >= 2])

    c = {"I0": _ary, "O": output}
    use_sections = Select([True, False], context=c)
    _axis = 0

    if use_sections:
        _indices_or_sections = Select(set(get_non_1_prime_factors(_ary.shape[_axis])))
    else:
        _indices_or_sections = Subset(range(_ary.shape[_axis]))

    return np.vsplit(_ary, _indices_or_sections), {
        "ary": _ary, "indices_or_sections": _indices_or_sections
    }


@generator(group='numpy', name='tile')
def gen_tile(inputs, output, **kwargs):
    _A = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])
    c = {"I0": _A, "O": output}

    _reps_n_dim = Select(range(1, MAX_REPS_N_DIM), context=c)
    _reps = []
    for i in range(_reps_n_dim):
        _reps.append(Select(range(MAX_REPS_PER_DIM), context=c))

    return np.tile(_A, _reps), {
        'A': _A, 'reps': _reps
    }


@generator(group='numpy', name='repeat')
def gen_repeat(inputs, output, **kwargs):
    #_a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])
    _a = inputs[0]
    c = {"I0": _a, "O": output}

    _repeats = Select(range(MAX_REPEATS), context=c)
    _axis = Select([None] + list(range(_a.ndim)), context=c)

    return np.repeat(_a, _repeats, axis=_axis), {
        'a': _a, 'repeats': _repeats, 'axis': _axis
    }

# Operators
@generator(group='numpy', name='ndarray.__equal__')
def gen_operator_equal(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}
    _b = Select(list(range(-5, 5)) + [np.max(_a), np.min(_a)], context=c)
    # todo(lmzheng): which value to choose?

    return _a == _b, {
        "lhs": _a, "rhs": _b
    }

# Others
#@generator(group='numpy', name='where')
#def gen_where(inputs, output, **kwargs):
#    _condition = Select([inp for inp in inputs if isinstance(inp, np.ndarray) and inp.dtype == np.bool])
#
#    _x = Select()

@generator(group='numpy', name='unique')
def gen_unique(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}
    _return_index = Select([True, False], context=c)
    _return_inverse = Select([True, False], context=c)
    _return_counts = Select([True, False], context=c)
    _axis = Select([None] + list(range(_a.ndim)), context=c)
    return np.unique(_a, return_index=_return_index, return_inverse=_return_inverse,
                     return_counts=_return_counts, axis=_axis), {
        "a": _a, "return_index": _return_index, "return_inverse": _return_inverse,
        "return_counts": _return_counts, "axis": _axis
    }


@generator(group='numpy', name='meshgrid')
def gen_meshgrid(inputs, output, **kwargs):
    c = wrap_io_context(inputs, output)

    #_xs = OrderedSubset([inp for inp in inputs if isinstance(inp, np.ndarray)])

    x1 = Select([inp for inp in inputs if isinstance(inp, np.ndarray)], context=c)
    x2 = Select([inp for inp in inputs if isinstance(inp, np.ndarray) if inp is not x1], context=c)
    _xs = (x1, x2)

    c = wrap_io_context(_xs, output)
    _indexing = "xy" #Select(["xy", "ij"], context=c)
    _sparse = False #Select([True, False], context=c)
    _copy = False #Select([True, False], context=c)

    return np.meshgrid(*_xs, indexing=_indexing, sparse=_sparse, copy=_copy), {
        "xs": _xs, "indexing": _indexing, "sparse": _sparse, "copy": _copy
    }


@generator(group='numpy', name='linalg.norm')
def gen_linalg_norm(inputs, output, **kwargs):
    _x = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _x, "O": output}
    _ord = Select([np.inf, -np.inf, 'fro', 'nuc', 1, 2], context=c)
    _axis = Select([None] + list(range(1, _x.ndim)), context=c)
    _keepdims = Select([True, False], context=c)

    prog = {"x": _x, "ord": _ord, "axis": _axis, "keepdims": _keepdims}
    return np.linalg.norm(**prog), prog


@generator(group='numpy', name='ndarray.__sub__')
def gen_ndarray_sub(inputs, output, **kwargs):
    lhs, rhs = OrderedSubset([inp for inp in inputs if isinstance(inp, np.ndarray)],
                             lengths=[2])

    return lhs - rhs, {"lhs": lhs, "rhs": rhs}


@generator(group='numpy', name='matmul')
def gen_matmul(inputs, output, **kwargs):
    _x1, _x2 = OrderedSubset([inp for inp in inputs if isinstance(inp, np.ndarray)],
                             lengths=[2])

    return np.matmul(_x1, _x2), {"x1": _x1, "x2": _x2}


@generator(group='numpy', name='sum')
def gen_sum(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}
    _axis = Select([None] + list(range(_a.ndim)), context=c)
    _out = None
    _dtype = Select(list(CAND_DTYPES.union([output.dtype] if isinstance(output, np.ndarray) else [])),
                    context=c)
    _keepdims = Select([True, False], context=c)

    prog = {"a": _a, "axis": _axis, "dtype": _dtype, "out": _out, "keepdims": _keepdims}
    return np.sum(**prog), prog


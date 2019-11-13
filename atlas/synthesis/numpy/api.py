import logging
from typing import Callable

import pandas as pd
import numpy as np

from ...generators import generator
from ...stubs import Select, Sequence, Subset, OrderedSubset, Product
from ..utils import wrap_io_context
from .utils import get_non_1_prime_factors

## Configs for random ndarray
MAX_N_DIM = 6         # Maximum number of dimensions
MIN_N_DIM = 1         # Minimum number of dimensions
MIN_DIM_LENGTH = 1    # Minimum length per dimension
MAX_DIM_LENGTH = 10   # Maximum length per dimension
CAND_DTYPES = set([np.dtype('float')])  #, np.dtype('int')])

# Configs for some APIs
MAX_REPEATS = 8       # max value for arg `repeats` in numpy.repeat
MAX_REPS_N_DIM = 5    # max number of dimensions in numpy.tile
MAX_REPS_PER_DIM = 5  # max tile number for each dimensions in numpy.tile

########### Array Creation Routines ##########
@generator(group='numpy', name='arange')
def gen_arange(inputs, output, **kwargs):
    c = {"O": output}
    _start = Select(range(MAX_DIM_LENGTH), context=c)
    to_add = {"I0": int(np.max(output)) + 1} if output is not None else {}
    _stop = Select(range(_start, MAX_DIM_LENGTH), context={**to_add, **c})
    _step = Select(range(1, _stop - _start + 1), context=c)
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

@generator(group='numpy', name='zeros')
def gen_zeros(inputs, output, **kwargs):
    c = {"O": output}

    n_dim = Select(range(1, MAX_N_DIM), context=c)

    shape_values = [1]
    for i in inputs:
        shape_values.extend(i.shape)
    if output is not None:
        shape_values.extend(output.shape)
    shape_values = list(set(shape_values))

    _shape = []
    for i in range(n_dim):
        _shape.append(Select(shape_values, context={"I": i, **c}))

    _dtype = Select(list(CAND_DTYPES.union([output.dtype] if isinstance(output, np.ndarray) else [])),
                    context=c)
    _order = "C" # Select(["C", "F"], context=c)

    return np.zeros(_shape, dtype=_dtype, order=_order), {
        "shape": _shape, "dtype": _dtype, "order": _order
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

########## Data movement ##########

@generator(group='numpy', name='reshape')
def gen_reshape(inputs, output, **kwargs):
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    c = {"I0": _a, "O": output}

    # How many dimensions does one want to reshape to?
    n_dims = Select(range(MIN_N_DIM, MAX_N_DIM), context=c)
    if n_dims == 1:
        _newshape = _a.size
    else:
        non_1_factors = get_non_1_prime_factors(_a.size)

        _newshape = []
        for i in range(n_dims - 1):
            factors_to_use = Subset(list(non_1_factors), include_empty=True, 
                                    context={"I1": i, **c})
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

    _order = SelectFixed(['C', 'F'], context=c)   # miss 'A'

    return np.reshape(_a, _newshape, order=_order), {
        'a': _a, 'newshape': _newshape, 'order': _order
    }


@generator(group='numpy', name='ndarray.flatten')
def gen_flatten(inputs, output, **kwargs):
    _self = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    try:
        if np.take(_self, 1) == np.take(output, 1):
            hint = 1
        else:
            hint = 0
    except IndexError:
        hint = 0

    c = {"I0": _self, "O": output, "H": np.ones((hint * 2,))}
    _order = SelectFixed(["C", "F"], context=c)

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
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    if isinstance(output, np.ndarray):
        hint = np.array([x for x in range(min(_a.ndim, output.ndim)) if
                _a.shape[x] != output.shape[x]])
    else:
        hint = None

    c = {"I0": _a, "H0": hint, "O": output}
    _axis1, _axis2 = OrderedSubset(range(_a.ndim), context=c, lengths=[2])

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
    domain = [inp for inp in inputs if isinstance(inp, np.ndarray) and inp.ndim <= 3]
    c = wrap_io_context(inputs, output)

    _tup = OrderedSubset(domain, context=c, lengths=[2])

    return np.dstack(_tup), {
        "tup": _tup
    }


@generator(group='numpy', name='hstack')
def gen_hstack(inputs, output, **kwargs):
    domain = [inp for inp in inputs if isinstance(inp, np.ndarray) and inp.ndim <= 3]
    c = wrap_io_context(inputs, output)

    _tup = OrderedSubset(domain, context=c, lengths=[2])

    return np.hstack(_tup), {
        "tup": _tup
    }


@generator(group='numpy', name='vstack')
def gen_vstack(inputs, output, **kwargs):
    domain = [inp for inp in inputs if isinstance(inp, np.ndarray) and inp.ndim <= 3]
    c = wrap_io_context(inputs, output)

    _tup = OrderedSubset(domain, context=c, lengths=[2])

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
    _A = SelectExternal(inputs, dtype=np.ndarray, **kwargs)
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
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)
    c = {"I0": _a, "O": output}

    _axis = Select([None] + list(range(_a.ndim)), context=c)
    _repeats = Select(range(MAX_REPEATS), context=c)

    return np.repeat(_a, _repeats, axis=_axis), {
        'a': _a, 'repeats': _repeats, 'axis': _axis
    }


@generator(group='numpy', name='flip')
def gen_flip(inputs, output, **kwargs):
    _m = SelectExternal(inputs, dtype=np.ndarray, **kwargs)
    c = {"I0": _m, "O": output}

    _axis = Select([None] + list(range(_m.ndim)), context=c)

    prog = {"m": _m, "axis": _axis}
    return np.flip(**prog), prog


# Primary Operators
@generator(group='numpy', name='ndarray.__equal__')
def gen_operator_equal(inputs, output, **kwargs):
    _a = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _a, "O": output}
    _b = Select(list(range(-5, 5)) + [np.max(_a), np.min(_a)], context=c)
    # todo(lmzheng): which value to choose?

    return _a == _b, {
        "lhs": _a, "rhs": _b
    }

@generator(group='numpy', name='ndarray.__sub__')
def gen_ndarray_sub(inputs, output, **kwargs):
    lhs = SelectExternal(inputs, dtype=(np.ndarray, float), **kwargs)
    rhs = SelectExternal(inputs, dtype=(np.ndarray, float), **kwargs)

    return lhs - rhs, {"lhs": lhs, "rhs": rhs}


@generator(group='numpy', name='ndarray.__div__')
def gen_ndarray_div(inputs, output, **kwargs):
    lhs = SelectExternal(inputs, dtype=(np.ndarray, float), **kwargs)
    rhs = SelectExternal(inputs, dtype=(np.ndarray, float), **kwargs)

    return lhs / rhs, {"lhs": lhs, "rhs": rhs}


@generator(group='numpy', name='logical_not')
def gen_logical_not(inputs, output, **kwargs):
    _x = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    prog = {"x": _x}
    return np.logical_not(_x), prog


@generator(group='numpy', name='matmul')
def gen_matmul(inputs, output, **kwargs):
    _x1 = SelectExternal(inputs, dtype=np.ndarray, **kwargs)
    _x2 = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    return np.matmul(_x1, _x2), {"x1": _x1, "x2": _x2}


### Indexing
@generator(group='numpy', name='ravel')
def gen_ravel(inputs, output, **kwargs):
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    c = {"I0": _a, "O": output}
    _order = Select(['C', 'F', 'A'], context=c)

    prog = {"a": _a, "order": _order}
    return np.ravel(*prog), prog


@generator(group='numpy', name='unravel_index')
def gen_unravel_index(inputs, output, **kwargs):
    valid_inputs = [x for x in inputs if (isinstance(x, np.ndarray) and "int" in str(x.dtype)) 
                                         or "int" in str(type(x))]
    _indices = SelectExternal(valid_inputs, **kwargs)

    c = {"I0": _indices, "O": output}
    shapes = [x.shape for x in inputs if isinstance(x, np.ndarray)]
    if isinstance(output, np.ndarray):
        shapes.append(output.shape)

    _shape = Select(shapes, context=c)

    prog = {"indices": _indices, "shape": _shape}
    return np.unravel_index(**prog), prog


@generator(group='numpy', name='take')
def gen_take(inputs, output, **kwargs):
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    c = {"I0": _a, "O": output}

    # todo(lmzheng): fix the case when indices is an input array. See also unravel_index
    _axis = Select([None] + list(range(_a.ndim)), context=c)

    max_len = _a.size if _axis is None else _a.shape[_axis]
    indices_domain = [x for x in inputs if ((isinstance(x, np.ndarray) and "int" in str(x.dtype)) 
                                           or "int" in str(type(x))) and np.max(x) < max_len]
    if _axis is not None:
        indices_domain.extend(range(_a.shape[_axis]))

    _indices = Select(indices_domain, context=c)

    prog = {"a": _a, "axis": _axis, "indices": _indices}
    return np.take(**prog), prog

@generator(group='numpy', name='ndarray.__getitem__')
def gen_ndarray_getitem_(inputs, output, **kwargs):
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    index = []
    for i in range(_a.ndim):
        c = {"I0": _a, "O": output, "I1": i}
        max_len = _a.shape[i]
        start = Select(range(max_len), context=c)
        stop = Select(range(start, max_len), context=c)
        if stop - start != 0:
            step = Select(range(1, stop - start + 1), context=c)
        else:
            step = 1
        index.append(slice(start, stop, step))

    index = tuple(index)
    prog = {"index": index}
    return _a[index], prog

@generator(group='numpy', name='compress')
def gen_compress(inputs, output, **kwargs):
    _condition = SelectExternal(inputs, dtype=np.ndarray, **kwargs)
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    # todo(lmzheng): use OrderedSet? 
    _axis = None

    prog = {"condition": _condition, "a": _a, "axis": _axis}
    return np.compress(**prog), prog

@generator(group='numpy', name='delete')
def gen_compress(inputs, output, **kwargs):
    _arr = SelectExternal(inputs, dtype=np.ndarray, **kwargs)
    _obj = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    # todo(lmzheng): use OrderedSet? 
    _axis = None

    prog = {"arr": _arr, "obj": _obj, "axis": _axis}
    return np.delete(**prog), prog


########## Linear algebra ##########
@generator(group='numpy', name='linalg.norm')
def gen_linalg_norm(inputs, output, **kwargs):
    _x = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    c = {"I0": _x, "O": output}
    #_ord = SelectFixed([None, np.inf, -np.inf, 0, 1, -1], context=c)
    _ord = SelectFixed([None], context=c)
    _axis = None # Select([None] + list(range(1, _x.ndim)), context=c)
    _keepdims = SelectFixed([True, False], context=c)

    prog = {"x": _x, "ord": _ord, "axis": _axis, "keepdims": _keepdims}
    return np.linalg.norm(**prog), prog


@generator(group='numpy', name='sum')
def gen_sum(inputs, output, **kwargs):
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    c = {"I0": _a, "O": output}
    _axis = Select([None] + list(range(_a.ndim)), context=c)
    _out = None
    _dtype = Select(list(CAND_DTYPES.union([output.dtype] if isinstance(output, np.ndarray) else [])),
                    context=c)
    _keepdims = Select([True, False], context=c)

    prog = {"a": _a, "axis": _axis, "dtype": _dtype, "out": _out, "keepdims": _keepdims}
    return np.sum(**prog), prog

# Others
#@generator(group='numpy', name='where')
#def gen_where(inputs, output, **kwargs):
#    _condition = Select([inp for inp in inputs if isinstance(inp, np.ndarray) and inp.dtype == np.bool])
#
#    _x = Select()


@generator(group='numpy', name='argmax')
def gen_argmax(inputs, output, **kwargs):
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    c = {"I0": _a, "O": output}
    _axis = Select([None] + list(range(_a.ndim)), context=c)

    prog = {"a": _a, "axis": _axis}
    return np.argmax(**prog), prog


@generator(group='numpy', name='argsort')
def gen_argsort(inputs, output, **kwargs):
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    c = {"I0": _a, "O": output}
    #  todo(lmzheng): fix this? disabled None to avoid array explosion
    #_axis = Select([None] + list(range(_a.ndim)), context=c) 
    _axis = Select(range(_a.ndim), context=c)

    prog = {"a": _a, "axis": _axis}
    return np.argsort(**prog), prog


@generator(group='numpy', name='searchsorted')
def gen_searchsorted(inputs, output, **kwargs):
    _a = SelectExternal(inputs, dtype=np.ndarray, **kwargs)
    _v = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    prog = {"a": _a, "v": _v}
    return np.searchsorted(**prog), prog

@generator(group='numpy', name='minimum')
def gen_argsort(inputs, output, **kwargs):
    _x1 = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    c = {"I": _x1, "O": output}
    _x2 = Select([x for x in inputs if isinstance(x, np.ndarray) and x is not _x1], context=c)

    prog = {"x1": _x1, "x2": _x2}
    return np.minimum(_x1, _x2), prog


@generator(group='numpy', name='unique')
def gen_unique(inputs, output, **kwargs):
    _ar = Select([inp for inp in inputs if isinstance(inp, np.ndarray)])

    c = {"I0": _ar, "O": output}
    # todo(lmzheng): Here the return values are two ndarrays with different shapes.
    #                How to handle this?
    _return_index = False # Select([True, False], context=c)
    _return_inverse = False # Select([True, False], context=c)
    _return_counts = Select([True, False], context=c)
    _axis = Select([None] + list(range(_ar.ndim)), context=c)
    prog = {"ar": _ar, "return_index": _return_index, "return_inverse": _return_inverse,
            "return_counts": _return_counts, "axis": _axis}
    return np.unique(**prog), prog


@generator(group='numpy', name='isnan')
def gen_isnan(inputs, output, **kwargs):
    _x = SelectExternal(inputs, dtype=np.ndarray, **kwargs)

    prog = {"x": _x}
    return np.isnan(_x), prog


@generator(group='numpy', name='meshgrid')
def gen_meshgrid(inputs, output, **kwargs):
    c = wrap_io_context(inputs, output)

    _xs = OrderedSubset([inp for inp in inputs if isinstance(inp, np.ndarray)], context=c, lengths=[2])

    c = wrap_io_context(_xs, output)
    _indexing = "xy" #Select(["xy", "ij"], context=c)
    _sparse = False #Select([True, False], context=c)
    _copy = False #Select([True, False], context=c)

    return np.meshgrid(*_xs, indexing=_indexing, sparse=_sparse, copy=_copy), {
        "xs": _xs, "indexing": _indexing, "sparse": _sparse, "copy": _copy
    }




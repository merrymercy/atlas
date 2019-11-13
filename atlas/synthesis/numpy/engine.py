import warnings
import traceback

import numpy as np

from ...exceptions import ExceptionAsContinue, StepTermination
from ...utils import get_group_by_name
from ...generators import generator 
from ..utils import wrap_io_context

numpy_apis =  {gen.name: gen for gen in get_group_by_name('numpy')}

@generator
def numpy_sequence_generator(inputs, output, funcs=[], max_seq_len=3):
    ctx = wrap_io_context(inputs, output)
    func_seq = SequenceFixed(funcs, max_len=max_seq_len, context=ctx, uid='func_seq')

    intermediates = list(inputs)
    unused_intermediates = set([id(x) for x in inputs])

    prog = []
    val = None
    for func_name in func_seq:
        func = numpy_apis[func_name]
        try:
            val, args = func(list(intermediates), output,
                             unused_intermediates=unused_intermediates)
        except StepTermination as e:
            raise e
        except Exception as e:
            warnings.warn(f"{func_name} :  {str(e)}")
            warnings.warn(f"{func_name} :  {traceback.format_exc()}")
            raise ExceptionAsContinue

        for x in args.values():
            unused_intermediates.discard(id(x))

        if isinstance(val, (list, tuple)):
            intermediates.extend(val)
            unused_intermediates.update([id(x) for x in val])
        else:
            intermediates.append(val)
            unused_intermediates.add(id(val))

        prog.append((func.name, args))

    if isinstance(val, np.ndarray):
        return val, prog
    elif isinstance(val, (tuple, list)): 
        if (all(isinstance(x, np.ndarray) for x in val) and 
            all(x.shape == val[0].shape for x in val)):
            # for functions that return tuples of array (e.g. meshgrid)
            return np.array(val), prog
        else:
            # only return np.ndarray.
            # This is a limitation of our encoder, which only supports np.ndarray
            # and does not support tuple of np.ndarray
            for x in val:
                if isinstance(x, np.ndarray):
                    return x, prog
        # list of int, object
        return np.array(val), prog
    else:
        return val, prog


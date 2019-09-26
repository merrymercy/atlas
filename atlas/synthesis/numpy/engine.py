import warnings
import traceback

from ...exceptions import ExceptionAsContinue
from ...utils import get_group_by_name
from ...generators import generator 
from ..utils import wrap_io_context

numpy_apis =  {gen.name: gen for gen in get_group_by_name('numpy')}

@generator
def numpy_sequence_generator(inputs, output, funcs=[], max_seq_len=3):
    ctx = wrap_io_context(inputs, output)
    func_seq = Sequence(funcs, max_len=max_seq_len, context=ctx, uid='func_seq')

    intermediates = []
    prog = []
    for func_name in func_seq:
        func = numpy_apis[func_name]
        try:
            val, args = func(intermediates + inputs, output)
        except Exception as e:
            warnings.warn(f"{func_name} :  {str(e)}")
            #warnings.warn(f"{func_name} :  {traceback.format_exc()}")
            raise ExceptionAsContinue

        if isinstance(val, (list, tuple)):
            intermediates.extend(val)
        else:
            intermediates.append(val)
        prog.append((func.name, args))

    return intermediates[-1], prog


import os, warnings
from typing import Collection, Any, Mapping, Callable

from atlas.utils.ioutils import IndexedFileReader, IndexedFileWriter
from ..tracing import OpTrace
import multiprocessing

global_encode_func = None
def encode_op(op):
    global global_encode_func

    return global_encode_func(domain=op.domain, context=op.context, 
                              choice=op.choice, sid=op.op_info.sid)

def dump_encodings(data: Collection[OpTrace], encode_func: Callable, path: str = None):
    global global_encode_func
    if isinstance(data, IndexedFileReader):
        if path is None:
            path = f"{data.path}.encoded"

        if os.path.exists(path):
            print(f"- Use cached encoding {path}")
            return IndexedFileReader(path)

    path = path or "data.encoded"

    global_encode_func = encode_func
    pool = multiprocessing.Pool()

    def patch_func(func, data):
        ret = []
        for x in data:
            ret.append(func(x))
        return ret
    pool.map = patch_func

    encoded_graphs = pool.map(encode_op, data)
    del pool

    encoding_file = IndexedFileWriter(path)
    for graph in encoded_graphs:
        encoding_file.append(graph)

    encoding_file.close()
    return IndexedFileReader(path)


def wrap_io_context(inputs: Collection[Any], output: Collection[Any]):
    c = {"O": output}
    for i, v in enumerate(inputs):
        c[f"I{i}"] = v
    return c


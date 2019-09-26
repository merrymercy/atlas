import os, warnings
from typing import Collection, Any, Mapping, Callable

from atlas.utils.ioutils import IndexedFileReader, IndexedFileWriter
from ..tracing import OpTrace


def dump_encodings(data: Collection[OpTrace], encode_func: Callable, path: str = None):
    if isinstance(data, IndexedFileReader):
        if path is None:
            path = f"{data.path}.encoded"

        if os.path.exists(path):
            return IndexedFileReader(path)

    path = path or "data.encoded"

    encoding_file = IndexedFileWriter(path)
    for op in data:
        encoding_file.append(encode_func(
            domain=op.domain,
            context=op.context,
            choice=op.choice,
            sid=op.op_info.sid
        ))

    encoding_file.close()
    return IndexedFileReader(path)


def wrap_io_context(inputs: Collection[Any], output: Collection[Any]):
    c = {"O": output}
    for i, v in enumerate(inputs):
        c[f"I{i}"] = v
    return c


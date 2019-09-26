import numpy as np

import atlas
from atlas import generator
from atlas.synthesis.utils import wrap_io_context
from atlas.synthesis.numpy import NumpyChecker, NumpyGraphEncoder

# test encode select


@generator(group='numpy', name='encoder_tester')
def encoder_tester(inputs, output, **kwargs):
    c = wrap_io_context(inputs, output)

    a = Select(range(4), context=c)

    return None, None


if __name__ == '__main__':
    encoder = NumpyGraphEncoder()

    inputs = [
        np.array([1])
    ]

    output = np.array([[20, 21]])

    val, trace = encoder_tester.generate(inputs, output).with_tracing().first()
    
    for op in trace.op_traces:
        encode_func = encoder.get_encoder(op.op_info.sid)

        ret = encode_func(
            domain=op.domain,
            context=op.context,
            choice=op.choice,
            sid=op.op_info.sid
        )

        print(ret['nodes'])




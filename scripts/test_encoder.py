import numpy as np

import atlas
from atlas import generator
from atlas.synthesis.utils import wrap_io_context
from atlas.synthesis.numpy import NumpyChecker, NumpyGraphEncoder
from atlas.synthesis.numpy.encoders import EdgeType, NodeFeature

from gen_data import random_ndarray

# test encode select

@generator(group='numpy', name='encoder_tester')
def encoder_tester(inputs, output, **kwargs):
    c = wrap_io_context(inputs, output)

    a = Select(range(3), context=c)

    return None, None


if __name__ == '__main__':
    encoder = NumpyGraphEncoder()

    inputs = [
        np.array([[10, 11], [12, 13]]),
    ]

    output = np.repeat(inputs[0], axis=1, repeats=3)

    val, trace = encoder_tester.generate(inputs, output).with_tracing().first()
    
    for op in trace.op_traces:
        encode_func = encoder.get_encoder(op.op_info.sid)

        ret = encode_func(
            domain=op.domain,
            context=op.context,
            choice=op.choice,
            sid=op.op_info.sid
        )

        nodes, edges = ret['nodes'], ret['edges']
        print(f"nodes: {len(nodes)}, edges: {len(edges)}")


        # nodes
        print(f"Domain: {ret['domain']}   Choice: {ret['choice']}")
        node_str = ""
        for i, n_feas in enumerate(nodes):
            converted = []
            for x in n_feas:
                if x >= NodeFeature.DIM_BEGIN.value:
                    converted.append(f"DIM_{x - NodeFeature.DIM_BEGIN.value}")
                else:
                    converted.append(str(NodeFeature(x)).split(".")[1])
            node_str += "%-30s" % ("%d %s" % (i, str(converted)))
            if (i+1) % 4 == 0:
                node_str += "\n"
        print(node_str)
        print("")

        # edges
        edge_str = ""
        for i, e_feas in enumerate(edges):
            src, fea, dst = e_feas
            edge_str += "%-25s" % ("%d->%d %s" % (src, dst, str(EdgeType(fea)).split('.')[1]))
            if (i+1) % 6 == 0:
                edge_str += "\n"
        print(edge_str)
        print("")



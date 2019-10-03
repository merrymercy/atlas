from typing import Any, List, Mapping, Collection
from enum import Enum, auto

import numpy as np

from ...operators import unpack_sid

class EdgeType(Enum):
    ADJ_UP = auto()
    ADJ_DOWN = auto()
    EQUALITY = auto()
    INNER_EQUALITY = auto()

    INDEX = auto()      # from index to cell
    INDEX_FOR = auto()  # from cell to index

    REPRESENTOR = auto()
    REPRESENTED = auto()

    ATTR = auto()       # from representor to attr
    ATTR_OF = auto()    # from attr to representor

    END = auto()

class NodeFeature(Enum):
    # DataType
    OBJECT = auto()
    FLOAT = auto()
    INT = auto()
    STR = auto()
    NAN = auto()

    # Source
    INPUT = auto()
    OUTPUT = auto()
    DOMAIN = auto()

    # Role
    REPRESENTOR = auto()
    ARRAY_ATTR = auto()

    DIM_BEGIN = auto()

    @staticmethod
    def type_to_feature(value):
        if value is None or np.isnan(value):
            return NodeFeature.NAN
        elif isinstance(value, (np.float64, np.float32, float)):
            return NodeFeature.FLOAT
        elif isinstance(value, (np.int64, np.int32, int)):
            return NodeFeature.INT
        elif isinstance(value, str):
            return NodeFeature.STR
        else:
            return NodeFeature.OBJECT


class NumpyGraphEncoder:
    def __init__(self):
        self.value_collections = dict()
        self.nodes = []
        self.edges = []
        self.max_dim = 10

    def get_num_node_features(self):
        return NodeFeature.DIM_BEGIN.value + self.max_dim

    def get_num_edge_types(self):
        return EdgeType.END.value

    def get_encoder(self, sid: str):
        unpacked = unpack_sid(sid)
        op_type, uid = unpacked.op_type, unpacked.uid
        if uid is not None and hasattr(self, f"{op_type}_{label}"):
            return getattr(self, f"{op_type}_{label}")

        return getattr(self, op_type)

    @staticmethod
    def label_to_base_feature(label):
        if label.startswith("I"):
            return [NodeFeature.INPUT.value]
        elif label.startswith("O"):
            return [NodeFeature.OUTPUT.value]
        elif label.startswith("D"):
            return [NodeFeature.DOMAIN.value]
        else:
            return []

    def add_equality_edges(self, value, src_node_id, inner_nodes=[]):
        if value in self.value_collections:
            for x in self.value_collections[value]:
                edge_type = EdgeType.INNER_EQUALITY if x in inner_nodes else EdgeType.EQUALITY
                self.edges.append([x, edge_type.value, src_node_id])
                self.edges.append([src_node_id, edge_type.value, x])
            self.value_collections[value].add(src_node_id)
        else:
            self.value_collections[value] = set([src_node_id])

    def encode_ndarray(self, array, label):
        base_fea = NumpyGraphEncoder.label_to_base_feature(label)

        # representor nodes
        self.nodes.append([NodeFeature.REPRESENTOR.value] + base_fea)
        repre_node = len(self.nodes) - 1

        # array attr : min, max, ...
        self.nodes.append([NodeFeature.ARRAY_ATTR.value] + base_fea)
        t = len(self.nodes) - 1
        self.edges.append([repre_node, EdgeType.ATTR.value, t])
        self.edges.append([t, EdgeType.ATTR_OF.value, repre_node])
        self.add_equality_edges(array.dtype, t)
        if array.size == 0:
            self.add_equality_edges(-np.inf, t)
            self.add_equality_edges(np.inf, t)
        else:
            self.add_equality_edges(np.max(array), t)
            self.add_equality_edges(np.min(array), t)

        # dim nodes
        dim_nodes = []
        for i, dim in enumerate(array.shape):
            tmp = []
            for _ in range(dim):
                self.nodes.append([NodeFeature.DIM_BEGIN.value + i] + base_fea)
                t = len(self.nodes)-1
                tmp.append(t)
                self.edges.append([repre_node, EdgeType.REPRESENTOR.value, t])
                self.edges.append([t, EdgeType.REPRESENTED.value, repre_node])

                self.add_equality_edges(i, t)

            dim_nodes.append(tmp)
        self.max_dim = max(self.max_dim, array.ndim)

        # value nodes
        dim_lens = []
        prod = 1
        for x in reversed(array.shape):
            dim_lens.append(prod)
            prod *= x
        dim_lens = list(reversed(dim_lens))
        value_nodes_begin = len(self.nodes)

        inner_nodes = set()
        for i in range(np.prod(array.shape)):
            idx = []
            t = i
            for j in range(array.ndim):
                idx.append(t // dim_lens[j])
                t %= dim_lens[j]
            idx = tuple(idx)

            value = array[idx]
            self.nodes.append([NodeFeature.type_to_feature(value).value] + base_fea)
            t = len(self.nodes) - 1

            # index edges
            for j in range(array.ndim):
                self.edges.append([dim_nodes[j][idx[j]], EdgeType.INDEX.value, t])
                self.edges.append([t, EdgeType.INDEX_FOR.value, dim_nodes[j][idx[j]]])

            # representor edges
            self.edges.append([repre_node, EdgeType.REPRESENTOR.value, t])
            self.edges.append([t, EdgeType.REPRESENTED.value, repre_node])

            # adjacent edges
            for j in range(array.ndim):
                adj_idx = list(idx)
                adj_idx[j] += 1
                if adj_idx[j] >= 0 and adj_idx[j] < array.shape[j]:
                    dst = sum([adj_idx[k] * dim_lens[k] for k in range(array.ndim)]) + value_nodes_begin
                    self.edges.append([dst, EdgeType.ADJ_DOWN.value, t])
                    self.edges.append([t, EdgeType.ADJ_UP.value, dst])

            # equality edge
            inner_nodes.add(t)
            self.add_equality_edges(value, t, inner_nodes)

        return repre_node

    def encode_value(self, value, label):
        base_fea = NumpyGraphEncoder.label_to_base_feature(label)

        # representor
        self.nodes.append([NodeFeature.REPRESENTOR.value] + base_fea)
        repre_node = len(self.nodes) - 1

        # value node
        self.nodes.append([NodeFeature.type_to_feature(value).value] + base_fea)
        t = len(self.nodes) - 1
        self.edges.append([repre_node, EdgeType.REPRESENTOR.value, t])
        self.edges.append([t, EdgeType.REPRESENTED.value, repre_node])
        self.add_equality_edges(value, t)

        return repre_node

    def encode(self, x, label=None):
        if isinstance(x, np.ndarray):
            return self.encode_ndarray(x, label)
        else:
            return self.encode_value(x, label)

    def Select(self, domain: List[Any], context: Mapping[str, Any], choice=None,
               mode='training', **kwargs):
        context = context or {}

        # init
        self.nodes = []
        self.edges = []
        self.value_collections = {}
        self.label_to_node_id = {}

        # create nodes
        for k, v in context.items():
            self.label_to_node_id[k] = self.encode(v, k)
        for i, v in enumerate(domain):
            self.label_to_node_id[f"D{i}"] = self.encode(v, f"D{i}")

        ret = {"nodes": self.nodes, "edges": remove_multiple_edges(self.edges),
               "domain": [self.label_to_node_id[f"D{i}"] for i in range(len(domain))]}

        if mode == 'training':
            if isinstance(choice, np.ndarray):
                found = False
                for i, x in enumerate(domain):
                    if x is choice:
                        choice_idx = i
                        found = True
                        break
                assert found, "InternalError: choice is not listed in domain"
            else:
                choice_idx = domain.index(choice)

            ret['choice'] = self.label_to_node_id[f"D{choice_idx}"]
        else:
            ret['mapping'] = {self.label_to_node_id[f"D{i}"]: v 
                              for i, v in enumerate(domain)}

        return ret

def remove_multiple_edges(edges):
    ret = []
    visited = set()
    for edge in edges:
        if (edge[0], edge[2]) in visited:
            continue

        visited.add((edge[0], edge[1]))
        ret.append(edge)

    return ret

"""
  {
    "nodes": [ [fea1, fea2], [fea4, fea5], ... ]
    "edges": [ [src, edge_type, dst], [src, edge_type, dst], ... ]
    "domain": [4, 9, 12]
    "choice": 4
  }
"""

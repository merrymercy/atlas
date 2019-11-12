import json
import types
from typing import Collection, Dict, Optional, Iterator

import numpy as np

from ...tracing import OpTrace
from ...operators import unpack_sid
from ...models import IndependentOperatorsModel, TrainableModel
from ...models.tensorflow.graphs.operators import GGNN, SelectGGNN, SelectFixedGGNN,\
    SequenceGGNN, SequenceFixedGGNN, OrderedSubsetGGNN, SubsetGGNN 
from ..utils import dump_encodings
from .encoders import NumpyGraphEncoder


class NumpyGeneratorModel(IndependentOperatorsModel):
    def __init__(self, work_dir: str = None, model_configs: Optional[Dict] ={}):
        self.model_configs = model_configs
        super().__init__(work_dir)
    
    @classmethod
    def load(cls, path: str, model_configs: Optional[Dict]={}):
        model = cls(path, model_configs)
        with open(f"{path}/model_list.json", "r") as f:
            model.modeled_sids = {k: f"{path}/{v}" for k, v in json.load(f).items()}

        model.load_models()

        return model

    def get_op_model(self, sid: str, dataset: Collection[OpTrace]) -> Optional[TrainableModel]:
        unpacked = unpack_sid(sid)
        op_type, uid = unpacked.op_type, unpacked.uid

        if uid is not None and hasattr(self, f"{op_type}_{uid}"):
            func = getattr(self, f"{op_type}_{uid}")
        elif hasattr(self, op_type):
            func = getattr(self, op_type)
        else:
            return None

        if op_type == 'SelectFixed':
            domain_size = max(len(x.domain) for x in dataset)
            return func(sid, domain_size)
        elif op_type == 'Sequence' or op_type == 'OrderedSubset':
            max_length  = max(len(x.choice) for x in dataset) + 1
            return func(sid, max_length)
        elif op_type == 'SequenceFixed':
            domain_size = max(len(x.domain) for x in dataset)
            max_length = max(len(x.choice) for x in dataset) + 1
            return func(sid, domain_size, max_length)
        else:
            return func(sid)

    def Select(self, sid: str):
        config = {
            'random_seed': 0,
            'learning_rate': 0.005,
            'clamp_gradient_norm': 1.0,
            'node_dimension': 100,
            'classifier_hidden_dims': [100],
            'batch_size': 512,
            'use_propagation_attention': True,
            'edge_msg_aggregation': 'avg',
            'residual_connections': {},
            'layer_timesteps': [1, 1, 1],
            'graph_rnn_cell': 'gru',
            'graph_rnn_activation': 'tanh',
            'edge_weight_dropout': 0.1,
        }
        config.update(self.model_configs.get('Select', {}))

        return NumpyOperatorModel(config, sid, SelectGGNN)

    def SelectFixed(self, sid: str, domain_size: int):
        config = {
            'random_seed': 0,
            'learning_rate': 0.005,
            'clamp_gradient_norm': 1.0,
            'node_dimension': 128,
            'classifier_hidden_dims': [128, 128],
            'batch_size': 512,
            'use_propagation_attention': True,
            'edge_msg_aggregation': 'avg',
            'residual_connections': {},
            'layer_timesteps': [1, 1, 1],
            'graph_rnn_cell': 'gru',
            'graph_rnn_activation': 'tanh',
            'edge_weight_dropout': 0.1,

            'domain_size': domain_size
        }
        config.update(self.model_configs.get('SelectFixed', {}))
        return NumpyOperatorModel(config, sid, SelectFixedGGNN)

    def Sequence(self, sid: str, max_length: int):
        config = {
            'random_seed': 0,
            'learning_rate': 0.005,
            'clamp_gradient_norm': 1.0,
            'node_dimension': 100,
            'classifier_hidden_dims': [100],
            'batch_size': 512,
            'use_propagation_attention': True,
            'edge_msg_aggregation': 'avg',
            'residual_connections': {},
            'layer_timesteps': [1, 1, 1],
            'graph_rnn_cell': 'gru',
            'graph_rnn_activation': 'tanh',
            'edge_weight_dropout': 0.1,

            'max_length': max_length
        }
        config.update(self.model_configs.get('Sequence', {}))

        return NumpyOperatorModel(config, sid, SequenceGGNN)

    def SequenceFixed(self, sid: str, domain_size: int, max_length: int):
        config = {
            'random_seed': 0,
            'learning_rate': 0.005,
            'clamp_gradient_norm': 1.0,
            'node_dimension': 100,
            'classifier_hidden_dims': [100],
            'batch_size': 512,
            'use_propagation_attention': True,
            'edge_msg_aggregation': 'avg',
            'residual_connections': {},
            'layer_timesteps': [1, 1, 1],
            'graph_rnn_cell': 'gru',
            'graph_rnn_activation': 'tanh',
            'edge_weight_dropout': 0.1,

            'domain_size': domain_size,
            'max_length': max_length,
        }
        config.update(self.model_configs.get('SequenceFixed', {}))

        return NumpyOperatorModel(config, sid, SequenceFixedGGNN)

    def OrderedSubset(self, sid: int, max_length: int):
        config = {
            'random_seed': 0,
            'learning_rate': 0.005,
            'clamp_gradient_norm': 1.0,
            'node_dimension': 100,
            'classifier_hidden_dims': [100],
            'batch_size': 512,
            'use_propagation_attention': True,
            'edge_msg_aggregation': 'avg',
            'residual_connections': {},
            'layer_timesteps': [1, 1, 1],
            'graph_rnn_cell': 'gru',
            'graph_rnn_activation': 'tanh',
            'edge_weight_dropout': 0.1,

            'max_length': max_length
        }
        config.update(self.model_configs.get('OrderedSubset', {}))

        return NumpyOperatorModel(config, sid, OrderedSubsetGGNN)

    def Subset(self, sid: str):
        config = {
            'random_seed': 0,
            'learning_rate': 0.005,
            'clamp_gradient_norm': 1.0,
            'node_dimension': 100,
            'classifier_hidden_dims': [100],
            'batch_size': 512,
            'use_propagation_attention': True,
            'edge_msg_aggregation': 'avg',
            'residual_connections': {},
            'layer_timesteps': [1, 1, 1],
            'graph_rnn_cell': 'gru',
            'graph_rnn_activation': 'tanh',
            'edge_weight_dropout': 0.1,
        }
        config.update(self.model_configs.get('Subset', {}))

        return NumpyOperatorModel(config, sid, SubsetGGNN)


class NumpyOperatorModel(SelectGGNN, SelectFixedGGNN, SequenceGGNN, SequenceFixedGGNN, 
        OrderedSubsetGGNN, SubsetGGNN):
    def __init__(self, params: Dict, sid: str, op_class: GGNN):
        self.encoder = NumpyGraphEncoder()
        self.sid = sid
        params.update({
            'num_node_features': self.encoder.get_num_node_features(),
            'num_edge_types': self.encoder.get_num_edge_types(),
        })

        self.op_class = op_class
        self.op_class.__init__(self, params)

        self.fix_beam_search_bounding()

    def train(self, train_data: Iterator, valid_data: Iterator, **kwargs):
        encoded_train = dump_encodings(train_data, self.encoder.get_encoder(self.sid))
        if valid_data is not None:
            encoded_valid = dump_encodings(valid_data, self.encoder.get_encoder(self.sid))
        else:
            encoded_valid = None

        self.op_class.train(self, encoded_train, encoded_valid, **kwargs)

    def infer(self, domain, context=None, return_prob=False, **kwargs):
        encoding = self.encoder.get_encoder(self.sid)(domain, context, mode='inference', sid=self.sid)
        preds = self.op_class.infer(self, [encoding])[0]
        if 'max_len' in kwargs:
            max_len = kwargs['max_len']
            preds = [x for x in preds if len(x[0]) <= max_len]
        if return_prob:
            return [x[0] for x in preds], [x[1] for x in preds]
        return [val for val, prob in sorted(preds, key=lambda x: -x[1])]

    @classmethod
    def load(cls, path):
        ret = super().load(path)
        ret.fix_beam_search_bounding()
        return ret

    def fix_beam_search_bounding(self):
        # fix for overloading beam_search in SequenceGGNN, SequenceFixedGGNN, ...
        if hasattr(self.op_class, 'beam_search'):
            self.beam_search = types.MethodType(
                    getattr(self.op_class, 'beam_search'), self)


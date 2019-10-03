from ...models import IndependentOperatorsModel
from ...models.tensorflow.graphs.operators import SelectGGNN
from ..utils import dump_encodings
from .encoders import NumpyGraphEncoder


class NumpyGeneratorModel(IndependentOperatorsModel):
    def __init__(self, model_configs={}):
        self.model_configs = model_configs
        super(NumpyGeneratorModel, self).__init__()

    def Select(self, sid):
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

        return NumpySelect(config, sid)


class NumpySelect(SelectGGNN):
    def __init__(self, params, sid):
        self.encoder = NumpyGraphEncoder()
        self.sid = sid
        params.update({
            'num_node_features': self.encoder.get_num_node_features(),
            'num_edge_types': self.encoder.get_num_edge_types(),
        })

        super().__init__(params)
 
    def train(self, train_data, valid_data, *args, **kwargs):
        encoded_train = dump_encodings(train_data, self.encoder.get_encoder(self.sid))
        if valid_data is not None:
            encoded_valid = dump_encodings(valid_data, self.encoder.get_encoder(self.sid))
        else:
            encoded_valid = None

        super().train(encoded_train, encoded_valid, *args, **kwargs)

    def infer(self, domain, context=None, sid='', **kwargs):
        encoding = self.encoder.get_encoder(self.sid)(domain, context, mode='inference', sid=sid)
        preds = super().infer([encoding])[0]
        return [val for val, prob in sorted(preds, key=lambda x: -x[1])]


import torch
import torch.nn as nn
from networks.network import Network
from networks.layers import make_layers


class Transformer(Network):
    def __init__(self, config, model_type="server"):
        super(Transformer, self).__init__(config, model_type)
        # Make layers.
        self.embedding_layers, self.encoder_layers, self.dense_layers = self._make_layers()

    def forward(self, x):
        if len(self.embedding_layers) > 0:
            x = self.embedding_layers(x)
        if len(self.encoder_layers) > 0:
            x = self.encoder_layers(x)
        if len(self.dense_layers) > 0:
            x = x.mean(dim=1)
            x = self.dense_layers(x)
        return x

    def _make_layers(self):
        embedding_layers = []
        encoder_layers = []
        dense_layers = []
        for layer_config in self.neural_network:
            layers = make_layers(layer_config["layer_type"], **layer_config["args"])
            if layer_config["layer_type"] == 'TransformerEmbedding':
                embedding_layers.append(layers)
            elif layer_config["layer_type"] == 'Dense':
                dense_layers.append(layers)
            else:
                encoder_layers.append(layers)
        return nn.Sequential(*embedding_layers), nn.Sequential(*encoder_layers), nn.Sequential(*dense_layers)

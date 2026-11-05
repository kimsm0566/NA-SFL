import torch
import torch.nn as nn
from networks.layers import make_layers
from networks.network import Network


class CNN(Network):
    def __init__(self, config, model_type="server"):
        super(CNN, self).__init__(config, model_type)
        # Make layers.
        self.convolutional_layers, self.dense_layers = self._make_layers()
        self._initialize_weights()

    def forward(self, x):
        if len(self.convolutional_layers) > 0:
            x = self.convolutional_layers(x)
        if len(self.dense_layers) > 0:
            x = torch.flatten(x, start_dim=1)
            x = self.dense_layers(x)
        return x

    def _make_layers(self):
        convolutional_layers = []
        dense_layers = []
        for layer_config in self.neural_network:
            layers = make_layers(layer_config["layer_type"], **layer_config["args"])
            if layer_config["layer_type"] == 'Dense':
                dense_layers.append(layers)
            else:
                convolutional_layers.append(layers)
        return nn.Sequential(*convolutional_layers), nn.Sequential(*dense_layers)

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, config, model_type="server"):
        """
        Build a neural network
        :param config: network config
        :param model_type: "complete": complete model; "server": server-side model; "client": client-side model.
        """
        super(Network, self).__init__()
        self.neural_network = config.NEURAL_NETWORK
        self.auxiliary_network = config.AUXILIARY_NETWORK
        self.auxiliary_classifier = config.MODEL_CONFIG.get_auxiliary_classifier(config.MODEL_NAME) \
            if model_type == 'profile' else None
        assert config.LAYER_NUM_ON_CLIENT <= len(self.neural_network), \
            "Layer number on client is larger than total layer number."
        # By default all layers are deployed on clients.
        self.layer_num_on_client = len(self.neural_network) if config.LAYER_NUM_ON_CLIENT == -1 \
            else config.LAYER_NUM_ON_CLIENT
        self.profile_num_on_client = config.PROFILE_NUM_ON_CLIENT
        self.model_type = model_type
        self.neural_network = self._get_neural_network()

    def forward(self, x):
        pass

    def _make_layers(self):
        pass

    def _initialize_weights(self):
        pass

    def _get_neural_network(self):
        if self.model_type == "server":
            neural_network = self.neural_network[self.layer_num_on_client:]
        elif self.model_type == "client":
            neural_network = self.neural_network[:self.layer_num_on_client]
        elif self.model_type == "complete":
            neural_network = self.neural_network
        elif self.model_type == "auxiliary_net" or self.model_type == "auxiliary_classifier":
            neural_network = self.auxiliary_network
        elif self.model_type == "profile":
            if self.profile_num_on_client == -1 or self.profile_num_on_client == len(self.neural_network):
                neural_network = self.neural_network
            else:
                neural_network = self.neural_network[:self.profile_num_on_client]
                neural_network.append(self.auxiliary_classifier)
        else:
            raise Exception('Parameter "type" can only be one of "complete", "server", "client", '
                            '"auxiliary_net" and "auxiliary_classifier".')
        return neural_network

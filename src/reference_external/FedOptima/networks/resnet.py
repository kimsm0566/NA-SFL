import torch.nn as nn
from networks.cnn import CNN


class ResNet(CNN):
    def __init__(self, config, model_type="server"):
        super(ResNet, self).__init__(config, model_type)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

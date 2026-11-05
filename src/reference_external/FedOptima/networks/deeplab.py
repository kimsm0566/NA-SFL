import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.fcn import FCN


class DeepLab(FCN):
    def __init__(self, config, model_type="server"):
        super(DeepLab, self).__init__(config, model_type)

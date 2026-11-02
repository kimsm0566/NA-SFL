"""Small supervised auxiliary head at an existing CIFAR ResNet boundary."""
from torch import nn


class AuxiliaryHead(nn.Sequential):
    def __init__(self, channels=128, num_classes=10):
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(channels, num_classes))

"""One CIFAR ResNet18 structure partitioned only after complete residual stages."""

from collections import OrderedDict

from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet


VALID_CUTS = (1, 2, 3)


def configure(model: nn.Module, cut: int | None = None, side: str = "full") -> None:
    if side not in ("full", "client", "server"):
        raise ValueError(f"unsupported model side: {side}")
    if side != "full" and cut not in VALID_CUTS:
        raise ValueError(f"ResNet18 cut must be one of {VALID_CUTS}")

    backbone = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
    names = ("conv1", "bn1", "relu")
    stem_length = len(names)
    stages = [
        (name, getattr(backbone, name))
        for name in names + ("layer1", "layer2", "layer3", "layer4", "avgpool")
    ]
    # The selected stem precedes layer1; boundaries keep whole BasicBlocks.
    if side == "client":
        stages = stages[:stem_length + cut]
    elif side == "server":
        stages = stages[stem_length + cut:]
    model.features = nn.Sequential(OrderedDict(stages))
    if side != "client":
        model.classifier = nn.Sequential(OrderedDict(fc=backbone.fc))
    model.cut = cut

"""Canonical full/client/server interfaces for the NA-SFL CIFAR ResNet18."""
from __future__ import annotations
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from models.resnet18 import configure


def _require_model(model_name):
    if model_name != 'resnet18_cifar':
        raise ValueError(f'unsupported NA-SFL model: {model_name}')


class FullModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        _require_model(model_name)
        configure(self)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(torch.flatten(self.features(inputs), 1))


class ClientModel(nn.Module):
    def __init__(self, model_name: str, cut: int):
        super().__init__()
        _require_model(model_name)
        configure(self, cut, 'client')

    def forward(self, inputs: Tensor) -> Tensor:
        return self.features(inputs)


class ServerModel(nn.Module):
    def __init__(self, model_name: str, cut: int):
        super().__init__()
        _require_model(model_name)
        configure(self, cut, 'server')

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(torch.flatten(self.features(inputs), 1))


def model_layout(model_name: str, cut: int) -> dict:
    _require_model(model_name)
    if type(cut) is not int or cut not in (1, 2, 3):
        raise ValueError('ResNet18 cut must be 1, 2, or 3')
    shapes = {1: [64, 32, 32], 2: [128, 16, 16], 3: [256, 8, 8]}
    return {'input_shape': [3, 32, 32], 'boundary_shape_per_sample': shapes[cut], 'num_classes': 10}


def canonical_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def load_matching_state(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    target = model.state_dict()
    missing = [name for name in target if name not in state]
    if missing:
        raise KeyError(f"state is missing target keys: {missing}")
    model.load_state_dict({name: state[name] for name in target}, strict=True)


def average_states(
    states: Sequence[Mapping[str, Tensor]], weights: Sequence[int | float]
) -> dict[str, Tensor]:
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must have the same nonzero length")
    numeric_weights = [float(weight) for weight in weights]
    if any(not torch.isfinite(torch.tensor(weight)) or weight <= 0 for weight in numeric_weights):
        raise ValueError("weights must be finite and positive")
    total_weight = sum(numeric_weights)
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise ValueError("all states must contain identical ordered keys")

    averaged = {}
    for key in keys:
        first = states[0][key]
        if torch.is_floating_point(first):
            accumulator = torch.zeros_like(first, dtype=torch.float64)
            for state, weight in zip(states, numeric_weights):
                accumulator.add_(state[key].to(dtype=torch.float64), alpha=weight)
            averaged[key] = (accumulator / total_weight).to(dtype=first.dtype)
        else:
            averaged[key] = torch.stack([state[key] for state in states]).max(dim=0).values
        averaged[key] = averaged[key].detach().clone()
    return averaged

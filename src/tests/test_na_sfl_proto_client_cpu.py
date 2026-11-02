"""Client graph contracts, isolated from the CUDA hardware boundary."""
import pytest
import torch
from torch import nn


def test_slow_client_updates_through_original_prototype_graph(monkeypatch):
    from utils import sflv2
    from utils import na_sfl_proto_client as client
    monkeypatch.setattr(sflv2, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(sflv2, '_timed', lambda device, fn: (fn(), 0.))
    monkeypatch.setattr(client, '_timed', lambda device, fn: (fn(), 0.))
    model = nn.Conv2d(1, 1, 1, bias=False)
    nn.init.ones_(model.weight)
    node = client.PrototypeClient(0, model, {'optimizer': 'adam', 'lr': .001}, prototype_mode=True)
    images = torch.tensor([1., 3., 5., 7.]).reshape(4, 1, 1, 1)
    begun = node.begin((1, 0, 0), images)
    upload = node.prepare_upload((1, 0, 0), begun, torch.tensor([1, 1, 3, 3]))
    assert upload['job_kind'] == 'PROTO_JOB'
    assert upload['activation'].flatten().tolist() == [2., 6.]
    assert upload['counts'].tolist() == [2, 2]
    assert upload['labels'].tolist() == [1, 3]
    assert not upload['activation'].requires_grad
    assert node.pending.activation.requires_grad
    node.complete((1, 0, 0), torch.ones_like(upload['activation']))
    assert model.weight.grad.item() == 8.
    assert model.weight.item() < 1 and node.pending is None and node.version == 1
    assert not hasattr(node, 'auxiliary_step') and node.auxiliary_head is None


def test_fast_client_upload_is_sd_without_separate_prototype(monkeypatch):
    from utils import sflv2
    from utils import na_sfl_proto_client as client
    monkeypatch.setattr(sflv2, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(sflv2, '_timed', lambda device, fn: (fn(), 0.))
    node = client.PrototypeClient(1, nn.Conv2d(1, 1, 1), {'optimizer': 'adam', 'lr': .001})
    begun = node.begin((1, 1, 0), torch.ones(2, 1, 1, 1))
    upload = node.prepare_upload((1, 1, 0), begun, torch.tensor([1, 2]))
    assert upload['job_kind'] == 'SD_JOB' and 'counts' not in upload
    assert upload['activation'].shape == (2, 1, 1, 1)
    assert node.pending.activation.shape == (2, 1, 1, 1)

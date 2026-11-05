"""Synthetic CPU contracts for retained Hybrid graphs; no CUDA execution."""
import copy

import pytest
import torch
from torch import nn


def hybrid_api():
    from utils import na_sfl_hybrid
    return na_sfl_hybrid


@pytest.mark.parametrize('batch,ratio,count', [(128, .1, 13), (56, .1, 6), (1, .01, 1), (56, 0, 0), (8, 1, 8)])
def test_sampling_count_is_half_up_and_does_not_touch_model_rng(batch, ratio, count):
    api = hybrid_api()
    state = torch.get_rng_state().clone()
    first = api.sample_indices(batch, ratio, 42, (1, 0, 3))
    assert api.sample_count(batch, ratio) == count
    assert first.dtype == torch.int32 and first.shape == (count,)
    assert torch.equal(first, api.sample_indices(batch, ratio, 42, (1, 0, 3)))
    assert first.unique().numel() == count
    assert count == 0 or (first.min() >= 0 and first.max() < batch)
    assert torch.equal(state, torch.get_rng_state())


def test_sampling_uses_independent_seed_and_job_identity():
    api = hybrid_api()
    indices = api.sample_indices(128, .1, 42, (1, 0, 0))
    assert not torch.equal(indices, api.sample_indices(128, .1, 43, (1, 0, 0)))
    assert not torch.equal(indices, api.sample_indices(128, .1, 42, (2, 0, 0)))


@pytest.mark.parametrize('batch,ratio,seed,job', [(0, .1, 42, (1, 0, 0)), (8, float('nan'), 42, (1, 0, 0)),
    (8, 1.1, 42, (1, 0, 0)), (8, -.1, 42, (1, 0, 0)), (8, .1, True, (1, 0, 0)),
    (8, .1, 42, (0, 0, 0))])
def test_sampling_rejects_invalid_arguments(batch, ratio, seed, job):
    with pytest.raises(ValueError):
        hybrid_api().sample_indices(batch, ratio, seed, job)


def make_client(monkeypatch, *, ratio=.25, slow=True):
    from utils import sflv2, na_sfl_proto_client
    api = hybrid_api()
    monkeypatch.setattr(sflv2, '_model_device', lambda model: torch.device('cpu'))
    for module in (sflv2, na_sfl_proto_client, api):
        monkeypatch.setattr(module, '_timed', lambda device, fn: (fn(), 0.))
    model = nn.Conv2d(1, 2, 1, bias=False)
    node = api.RandomSDClient(0, model, {'optimizer': 'sgd', 'lr': .01,
        'random_sd': {'sample_ratio': ratio, 'sample_seed': 42}}, prototype_mode=slow)
    return node


def prepare(node):
    images = torch.arange(1., 9.).reshape(8, 1, 1, 1)
    labels = torch.tensor([1, 1, 1, 1, 3, 3, 3, 3])
    begun = node.begin((1, 0, 0), images)
    return images, labels, node.prepare_upload((1, 0, 0), begun, labels)


def response(upload):
    return {'grad_selected_sd': torch.full_like(upload['selected_sd'], .8),
        'grad_prototypes': torch.full_like(upload['activation'], .2),
        'selected_indices': upload['selected_indices'].clone(), 'classes': upload['labels'].clone()}


def test_same_client_forward_two_branch_gradients_match_local_reference(monkeypatch):
    node = make_client(monkeypatch)
    reference = copy.deepcopy(node.model)
    calls = []
    node.model.register_forward_hook(lambda *args: calls.append(1))
    images, labels, upload = prepare(node)
    assert upload['job_kind'] == 'HYBRID_JOB' and upload['counts'].tolist() == [4, 4]
    assert upload['actual_batch_size'] == 8 and upload['selected_sd'].shape[0] == 2
    z = reference(images)
    p = torch.stack([z[labels == label].mean(0) for label in upload['labels']])
    g = response(upload)
    objective = (z[upload['selected_indices'].long()] * .8).sum() + (p * .2).sum()
    objective.backward()
    node.complete((1, 0, 0), g)
    torch.testing.assert_close(node.model.weight.grad, reference.weight.grad)
    assert len(calls) == 1 and node.version == 1 and node.pending is None
    assert node._hybrid_pending is None


def test_gradient_masking_combines_selected_and_full_batch_prototype_paths(monkeypatch):
    node = make_client(monkeypatch)
    images, labels, upload = prepare(node)
    z = node.pending.activation
    z.retain_grad()
    node.complete((1, 0, 0), response(upload))
    expected = torch.full_like(z, .2 / 4)
    expected[upload['selected_indices'].long()] += .8
    torch.testing.assert_close(z.grad, expected)


@pytest.mark.parametrize('bad', ['nan', 'shape', 'indices', 'classes', 'missing'])
def test_invalid_response_preserves_graph_and_model_for_correct_completion(monkeypatch, bad):
    node = make_client(monkeypatch)
    _, _, upload = prepare(node)
    good = response(upload)
    invalid = {key: value.clone() for key, value in good.items()}
    before = node.model.weight.detach().clone()
    if bad == 'nan': invalid['grad_prototypes'][0] = float('nan')
    if bad == 'shape': invalid['grad_selected_sd'] = invalid['grad_selected_sd'][:1]
    if bad == 'indices': invalid['selected_indices'] = invalid['selected_indices'].flip(0)
    if bad == 'classes': invalid['classes'] = invalid['classes'].flip(0)
    if bad == 'missing': invalid.pop('classes')
    with pytest.raises(ValueError): node.complete((1, 0, 0), invalid)
    assert node.version == 0 and node.pending is not None
    assert torch.equal(node.model.weight, before) and node.model.weight.grad is None
    node.complete((1, 0, 0), good)


@pytest.mark.parametrize('slow,ratio,kind', [(False, .1, 'SD_JOB'), (True, 0., 'PROTO_JOB')])
def test_fast_and_zero_ratio_reuse_old_upload_and_tensor_gradient(monkeypatch, slow, ratio, kind):
    node = make_client(monkeypatch, ratio=ratio, slow=slow)
    _, _, upload = prepare(node)
    assert upload['job_kind'] == kind and 'selected_sd' not in upload
    node.complete((1, 0, 0), torch.ones_like(upload['activation']))
    assert node.version == 1 and node.pending is None


def test_abort_clears_all_retained_graph_references(monkeypatch):
    node = make_client(monkeypatch)
    prepare(node)
    node.clear_pending()
    assert node.pending is None and node._hybrid_pending is None

"""Synthetic retained-graph and bookkeeping contracts for the SD-only ablation."""
import copy
import pytest
import torch
from torch import nn


def cfg(**changes):
    from utils.na_sfl_config import load_config
    return load_config(overrides={'method': 'na_sfl_random_sd_only', **changes})


def test_sd_only_config_and_cli_preserve_original_v5():
    from utils.na_sfl_config import load_config
    from utils.option import parse_args
    c = cfg()
    assert c['random_sd'] == {'sample_ratio': .1, 'sample_seed': 42}
    assert c['aux_client_count'] == 0 and c['bn_policy'] == 'shared_original'
    assert parse_args(['--method', c['method']]).config['method'] == c['method']
    old = load_config(overrides={'method': 'na_sfl_proto_random_sd'})
    assert old['bn_policy'] == 'dual_running_stats_shared_affine'
    assert old['prototype']['fast_sd_weight'] == .8 and old['prototype']['fast_proto_weight'] == .2


@pytest.mark.parametrize('changes', [
    {'random_sd': {'sample_ratio': 0}}, {'random_sd': {'sample_ratio': 1.1}},
    {'random_sd': {'sample_ratio': True}}, {'random_sd': {'sample_seed': -1}},
    {'slow_client_ids': [0, 0]}, {'aux_client_count': 1},
    {'client_compute_seconds': {'auxiliary': .1}},
    {'bn_policy': 'dual_running_stats_shared_affine'},
])
def test_invalid_sd_only_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        cfg(**changes)


def client(monkeypatch, slow=True):
    from utils import sflv2, na_sfl_hybrid
    monkeypatch.setattr(sflv2, '_model_device', lambda _: torch.device('cpu'))
    monkeypatch.setattr(sflv2, '_timed', lambda _, fn: (fn(), 0.))
    monkeypatch.setattr(na_sfl_hybrid, '_timed', lambda _, fn: (fn(), 0.))
    return na_sfl_hybrid.SDOnlyClient(0, nn.Conv2d(1, 2, 1, bias=False),
        {'optimizer': 'sgd', 'lr': .01, 'random_sd': {'sample_ratio': .1, 'sample_seed': 42}},
        sampled_mode=slow)


@pytest.mark.parametrize('batch,slow,expected', [(128, True, 13), (56, True, 6), (56, False, 56)])
def test_selected_labels_and_one_retained_forward_match_v5_indices(monkeypatch, batch, slow, expected):
    from utils.na_sfl_hybrid import sample_indices
    node = client(monkeypatch, slow)
    images = torch.arange(float(batch)).reshape(batch, 1, 1, 1)
    labels = torch.arange(batch) % 10
    calls = []
    node.model.register_forward_hook(lambda *a: calls.append(1))
    job = (1, 0, 0)
    begun = node.begin(job, images)
    whole = node.pending.activation
    whole.retain_grad()
    upload = node.prepare_sd_upload(job, begun, labels)
    indices = sample_indices(batch, .1, 42, job).long() if slow else torch.arange(batch)
    assert upload['selected_indices'].tolist() == indices.tolist()
    assert torch.equal(upload['labels'], labels[indices])
    assert upload['activation'].shape[0] == expected
    assert 'counts' not in upload and 'prototypes' not in upload and 'job_kind' not in upload
    node.complete(job, torch.ones_like(upload['activation']))
    expected_grad = torch.zeros_like(whole)
    expected_grad[indices] = 1.
    torch.testing.assert_close(whole.grad, expected_grad)
    assert calls == [1] and node.pending is None and node.version == 1
    assert node.auxiliary_head is None


def test_sd_only_rejects_duplicate_prepare_and_bad_gradient_without_updating(monkeypatch):
    node = client(monkeypatch)
    job = (1, 0, 0)
    begun = node.begin(job, torch.ones(56, 1, 1, 1))
    upload = node.prepare_sd_upload(job, begun, torch.arange(56) % 10)
    before = copy.deepcopy(node.model.state_dict())
    with pytest.raises(RuntimeError):
        node.prepare_sd_upload(job, begun, torch.arange(56) % 10)
    with pytest.raises(ValueError):
        node.complete(job, torch.ones(56, 2, 1, 1))
    assert node.version == 0 and node.pending is not None
    for key, value in before.items():
        assert torch.equal(value, node.model.state_dict()[key])
    node.complete(job, torch.ones_like(upload['activation']))


def test_fifo_capacity_uses_transmitted_sd_count():
    from collections import deque
    from utils.na_sfl_batch import take_ready_batch
    queue = deque([{'row': {'samples': 128, 'server_samples': 13}},
                   {'row': {'samples': 128, 'server_samples': 13}}])
    assert len(take_ready_batch(queue, 26, 'concat')) == 2


def test_batch_validation_counts_server_sd_separately_from_client_visits():
    from utils.na_sfl_batch import validate_server_batches
    job = dict(job_id=[1, 0, 0], round=1, samples=128, server_samples=13,
               server_batch_id=0, server_version_before=0, server_version_after=1)
    batch = dict(round=1, sd_count=1, samples=13, job_ids=[[1, 0, 0]], batch_id=0,
                 server_version_before=0, server_version_after=1)
    assert validate_server_batches([job], [batch], 512, 'individual') == 1
    with pytest.raises(ValueError):
        validate_server_batches([{**job, 'server_samples': 12}], [batch], 512, 'individual')


def test_sd_only_single_ce_server_gradient_matches_reference_without_any_prototype(monkeypatch):
    from utils import sflv2, na_sfl_hybrid, na_sfl_prototype
    from torch.nn import functional as F
    monkeypatch.setattr(sflv2, '_model_device', lambda _: torch.device('cpu'))
    monkeypatch.setattr(sflv2, '_timed', lambda _, fn: (fn(), 0.))
    monkeypatch.setattr(na_sfl_hybrid, '_timed', lambda _, fn: (fn(), 0.))
    def forbidden(*a, **k):
        raise AssertionError('Prototype computation must not run')
    monkeypatch.setattr(na_sfl_prototype, 'make_local_prototypes', forbidden)
    bottom = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
    top = nn.Sequential(nn.Conv2d(2, 3, 1), nn.BatchNorm2d(3), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    ref_bottom, ref_top = copy.deepcopy(bottom), copy.deepcopy(top)
    c = {'optimizer': 'sgd', 'lr': .01, 'num_classes': 3,
         'random_sd': {'sample_ratio': .1, 'sample_seed': 42}}
    node = na_sfl_hybrid.SDOnlyClient(0, bottom, c, sampled_mode=True)
    server = sflv2.SFLV2Server(top, c)
    x, y, job = torch.randn(56, 1, 2, 2), torch.arange(56) % 3, (1, 0, 0)
    upload = node.prepare_sd_upload(job, node.begin(job, x), y)
    indices = upload['selected_indices'].long()
    loss = F.cross_entropy(ref_top(ref_bottom(x)[indices]), y[indices])
    loss.backward()
    result = server.process_batch([{'job_id': job, 'activation': upload['activation'],
        'labels': upload['labels'], 'return_cut_gradient': True}])
    node.complete(job, result['jobs'][0]['cut_gradient'])
    assert result['jobs'][0]['loss'] == pytest.approx(loss.item())
    assert result['samples'] == 6 and server.version == node.version == 1
    for a, b in zip(bottom.parameters(), ref_bottom.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    for a, b in zip(top.parameters(), ref_top.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    assert top[1].num_batches_tracked.item() == 1


def test_sd_only_supervisor_requires_exact_v5_comparison_config():
    from utils.na_sfl_config import load_config, DEFAULTS
    from utils.na_sfl_sd_only_experiment import validate_config
    baseline = load_config(overrides={'method': 'na_sfl_proto_random_sd', 'rounds': 100})
    actual = cfg(rounds=100)
    validate_config(baseline, actual)
    with pytest.raises(ValueError):
        validate_config(baseline, cfg(rounds=100, lr=.002))
    with pytest.raises(ValueError):
        validate_config(baseline, {**actual, 'random_sd': {'sample_ratio': .2, 'sample_seed': 42}})


def test_failed_sd_only_smoke_prevents_main_launch(monkeypatch, tmp_path):
    from utils import na_sfl_sd_only_experiment as experiment
    calls = []
    def launch(name, rounds, smoke):
        calls.append(name)
        raise RuntimeError('smoke audit rejected')
    with pytest.raises(RuntimeError, match='smoke audit rejected'):
        experiment.launch_sequence(launch)
    assert calls == ['smoke_test']

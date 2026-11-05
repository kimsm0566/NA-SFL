"""Small synthetic BN contracts; never run ResNet or a GPU regression suite."""
import copy
import io

import pytest
import torch
from torch import nn


def _api():
    from utils.na_sfl_bn import ServerBatchNorm
    return ServerBatchNorm


def _model():
    return nn.Sequential(nn.Conv2d(2, 3, 1), nn.BatchNorm2d(3), nn.ReLU(),
                         nn.Dropout(.2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))


def _buffers(model):
    return {k: v.clone() for k, v in model.named_buffers()}


def _equal(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[k], right[k]) for k in left)


def test_freeze_only_bn_restores_modes_and_keeps_input_affine_and_model_gradients():
    model = _model().train()
    bn = _api()(model, 'sd_train_proto_eval_bn_only')
    with bn.domain('sd'):
        model(torch.randn(6, 2, 3, 3))
    before = _buffers(model)
    x = torch.randn(3, 2, 3, 3, requires_grad=True)
    with bn.domain('proto'):
        assert not model[1].training and model[3].training and model.training
        loss = nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2]))
    assert model[1].training and model[3].training
    loss.backward()
    _equal(before, _buffers(model))
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in [model[0].weight, model[1].weight, model[1].bias, model[-1].weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    with bn.domain('sd'):
        model(torch.randn(4, 2, 3, 3))
    assert model[1].num_batches_tracked.item() == 2
    assert bn.summary()['forward_calls'] == {'sd': 2, 'proto': 1}


def test_freeze_proto_before_first_sd_is_recorded_without_reordering():
    model = _model().train()
    bn = _api()(model, 'sd_train_proto_eval_bn_only')
    before = _buffers(model)
    with bn.domain('proto'):
        model(torch.randn(3, 2, 3, 3)).sum().backward()
    _equal(before, _buffers(model))
    assert bn.summary()['proto_forwards_before_first_sd'] == 1


@pytest.mark.parametrize('policy', ['sd_train_proto_eval_bn_only', 'dual_running_stats_shared_affine'])
def test_forward_exception_restores_bn_modes_and_sd_buffer_references(policy):
    model = _model().train()
    bn = _api()(model, policy)
    references = [model[1].running_mean, model[1].running_var, model[1].num_batches_tracked]
    with pytest.raises(RuntimeError, match='deliberate'):
        with bn.domain('proto'):
            raise RuntimeError('deliberate')
    assert model[1].training
    assert all(left is right for left, right in zip(references,
        [model[1].running_mean, model[1].running_var, model[1].num_batches_tracked]))
    bn.require_sd_domain()


def test_dual_isolates_statistics_shares_all_parameters_and_preserves_base_state_keys():
    model = _model().train()
    ids = [id(p) for p in model.parameters()]
    original = copy.deepcopy(model.state_dict())
    bn = _api()(model, 'dual_running_stats_shared_affine')
    _equal(original, model.state_dict())
    sd_before = _buffers(model)
    with bn.domain('proto'):
        proto_loss = model(torch.randn(3, 2, 3, 3) + 4).square().sum()
    _equal(sd_before, _buffers(model))
    proto_after = bn.state_dict()['layers']['1']['proto']
    assert proto_after['num_batches_tracked'].item() == 1
    with bn.domain('sd'):
        sd_loss = model(torch.randn(6, 2, 3, 3) - 3).square().sum()
    _equal(proto_after, bn.state_dict()['layers']['1']['proto'])
    (proto_loss + sd_loss).backward()
    assert ids == [id(p) for p in model.parameters()]
    assert model[1].weight.grad is not None and model[1].bias.grad is not None
    assert model[1].num_batches_tracked.item() == 1
    assert not torch.equal(model[1].running_mean, proto_after['running_mean'])


def test_dual_checkpoint_restores_both_domains_optimizer_and_exact_sd_eval_output():
    torch.manual_seed(73)
    model = _model().train()
    bn = _api()(model, 'dual_running_stats_shared_affine')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    losses = []
    for domain in ['sd', 'proto', 'sd']:
        with bn.domain(domain):
            losses.append(model(torch.randn(4, 2, 3, 3)).square().sum())
    sum(losses).backward(); optimizer.step()
    image = torch.randn(4, 2, 3, 3)
    model.eval(); expected = model(image).detach()
    stream = io.BytesIO()
    torch.save({'model': model.state_dict(), 'bn': bn.state_dict(), 'optimizer': optimizer.state_dict()}, stream)
    stream.seek(0); saved = torch.load(stream, weights_only=False)
    restored = _model(); restored_bn = _api()(restored, 'dual_running_stats_shared_affine')
    restored.load_state_dict(saved['model']); restored_bn.load_state_dict(saved['bn'])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
    restored_optimizer.load_state_dict(saved['optimizer'])
    restored.eval(); restored_bn.require_sd_domain()
    assert torch.equal(expected, restored(image))
    for domain in ['sd', 'proto']:
        _equal(saved['bn']['layers']['1'][domain], restored_bn.state_dict()['layers']['1'][domain])
    assert len(restored_optimizer.state) == len(optimizer.state)


def test_shared_policy_is_bitwise_identical_to_original_forward_backward_and_adam_trace():
    model = _model(); model[3] = nn.Identity()
    reference = copy.deepcopy(model)
    bn = _api()(model, 'shared_original')
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    ref_opt = torch.optim.Adam(reference.parameters(), lr=.001)
    for _ in range(3):
        sd, proto = torch.randn(8, 2, 3, 3), torch.randn(3, 2, 3, 3)
        opt.zero_grad(); ref_opt.zero_grad()
        with bn.domain('sd'): left = model(sd).square().mean()
        with bn.domain('proto'): left = .8 * left + .2 * model(proto).square().mean()
        right = .8 * reference(sd).square().mean() + .2 * reference(proto).square().mean()
        assert torch.equal(left, right)
        left.backward(); right.backward(); opt.step(); ref_opt.step()
        _equal(model.state_dict(), reference.state_dict())


@pytest.mark.parametrize('corruption', ['variance', 'policy', 'missing_domain', 'counter', 'positive_counter', 'sd_conflict'])
def test_reject_malformed_dual_checkpoint(corruption):
    model = _model(); bn = _api()(model, 'dual_running_stats_shared_affine')
    state = bn.state_dict()
    if corruption == 'variance': state['layers']['1']['proto']['running_var'][0] = -1
    if corruption == 'policy': state['policy'] = 'shared_original'
    if corruption == 'missing_domain': del state['layers']['1']['proto']
    if corruption == 'counter': state['layers']['1']['proto']['num_batches_tracked'].fill_(-1)
    if corruption == 'positive_counter': state['layers']['1']['proto']['num_batches_tracked'].fill_(58)
    if corruption == 'sd_conflict': state['layers']['1']['sd']['running_mean'].fill_(100)
    with pytest.raises(ValueError): bn.load_state_dict(state)


@pytest.mark.parametrize('policy,expected_sd,expected_proto', [
    ('shared_original', 3, None), ('sd_train_proto_eval_bn_only', 1, None),
    ('dual_running_stats_shared_affine', 1, 2)])
def test_prototype_server_combined_backward_one_step_and_domain_checkpoint_smoke(
        monkeypatch, policy, expected_sd, expected_proto):
    from utils import sflv2, na_sfl_prototype as api
    monkeypatch.setattr(sflv2, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(api, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(api, '_timed', lambda device, operation: (operation(), 0.))
    cfg = {'optimizer': 'adam', 'lr': .001, 'weight_decay': .0005, 'num_classes': 3,
        'method': 'na_sfl_v3_proto', 'boundary_shape_per_sample': [2, 3, 3], 'bn_policy': policy,
        'prototype': {'fast_sd_weight': .8, 'fast_proto_weight': .2, 'slow_proto_weight': 1.,
                      'alignment_weight_beta': .1, 'alignment_reduction': 'mse'}}
    model = _model(); model[3] = nn.Identity()
    server = api.PrototypeServer(model, cfg); server.start_round(1)
    slow = {'job_id': (1, 0, 0), 'job_kind': 'PROTO_JOB', 'activation': torch.randn(2, 2, 3, 3),
            'labels': torch.tensor([0, 2]), 'counts': torch.tensor([2, 3]),
            'actual_batch_size': 5, 'return_cut_gradient': True}
    first = server.process_batch([slow])['jobs'][0]
    fast = {'job_id': (1, 1, 0), 'job_kind': 'SD_JOB', 'activation': torch.randn(5, 2, 3, 3),
            'labels': torch.tensor([0, 0, 2, 2, 2]), 'actual_batch_size': 5, 'return_cut_gradient': True}
    item = server.process_batch([fast])['jobs'][0]
    assert first['bn_proto_before_first_sd'] == 1
    assert item['L_fast_weighted'] == pytest.approx(.8 * item['L_SD'] + .2 * item['L_proto'])
    assert item['cut_gradient'].shape == fast['activation'].shape
    assert torch.isfinite(item['cut_gradient']).all()
    assert server.version == 2 and model[1].num_batches_tracked.item() == expected_sd
    assert all(value['step'].item() == 2 for value in server.optimizer.state.values())
    state = server.extra_state_dict()
    if expected_proto is not None:
        assert state['batchnorm']['layers']['1']['proto']['num_batches_tracked'].item() == expected_proto
    fresh_model = copy.deepcopy(model)
    fresh = api.PrototypeServer(fresh_model, cfg)
    fresh.load_extra_state_dict(state)
    assert fresh.batchnorm.summary() == server.batchnorm.summary()


def test_checkpoint_bn_counters_must_match_recorded_jobs():
    from utils.na_sfl_bn import validate_bn_checkpoint_boundary
    model = _model(); bn = _api()(model, 'dual_running_stats_shared_affine')
    saved = {'config': {'bn_policy': 'dual_running_stats_shared_affine'},
             'server_extra_state': {'batchnorm': bn.state_dict()}, 'server_model': model.state_dict(),
             'job_records': [{'job_kind': 'SD_JOB', 'server_pseudo_forward_calls': 0}]}
    with pytest.raises(ValueError): validate_bn_checkpoint_boundary(saved)

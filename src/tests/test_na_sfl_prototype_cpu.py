"""CPU mathematical contracts; replace CUDA boundaries only, never train ResNet."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F


def _api():
    from utils import na_sfl_prototype
    return na_sfl_prototype


def _config(**prototype):
    return {'optimizer': 'adam', 'lr': .001, 'weight_decay': 0., 'num_classes': 3,
        'method': 'na_sfl_v3_proto', 'boundary_shape_per_sample': [2, 2, 2],
        'prototype': {'fast_sd_weight': .5, 'fast_proto_weight': .5,
            'slow_proto_weight': 1., 'alignment_weight_beta': .1,
            'alignment_reduction': 'mse', **prototype}}


def _top(bn=False):
    torch.manual_seed(11)
    return nn.Sequential(nn.Conv2d(2, 3, 1),
        *([nn.BatchNorm2d(3)] if bn else []), nn.AdaptiveAvgPool2d(1), nn.Flatten())


@pytest.fixture
def cpu_boundary(monkeypatch):
    from utils import sflv2
    api = _api()
    monkeypatch.setattr(sflv2, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(api, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(api, '_timed', lambda device, operation: (operation(), 0.))


def _packet(x, y, kind='SD_JOB', counts=None, job=(1, 0, 0)):
    packet = {'job_id': job, 'activation': x, 'labels': y,
        'return_cut_gradient': True, 'job_kind': kind,
        'actual_batch_size': len(y) if counts is None else int(counts.sum())}
    if counts is not None:
        packet['counts'] = counts
    return packet


def test_partial56_present_sorted_classes_and_exact_class_mean_chain_rule():
    api = _api()
    z = torch.arange(56 * 8, dtype=torch.float32).reshape(56, 2, 2, 2).requires_grad_()
    y = torch.tensor([2] * 40 + [0] * 16)
    p, classes, counts = api.make_local_prototypes(z, y)
    assert classes.tolist() == [0, 2] and counts.tolist() == [16, 40]
    assert torch.equal(p[0], z[40:].mean(0)) and torch.equal(p[1], z[:40].mean(0))
    upstream = torch.stack([torch.ones_like(p[0]) * 4, torch.ones_like(p[1]) * 8])
    p.backward(upstream)
    assert torch.equal(z.grad[40:], upstream[0].expand_as(z.grad[40:]) / 16)
    assert torch.equal(z.grad[:40], upstream[1].expand_as(z.grad[:40]) / 40)
    api.validate_prototypes(p, classes, counts, (2, 2, 2), 3, 56)


@pytest.mark.parametrize('bad', ['count', 'order', 'dtype', 'nan', 'shape'])
def test_prototype_validation_rejects_invalid_wire_values(bad):
    api = _api()
    p, c, n = torch.ones(2, 2, 2, 2), torch.tensor([0, 2]), torch.tensor([2, 3])
    if bad == 'count': n[0] = 0
    if bad == 'order': c = torch.tensor([2, 0])
    if bad == 'dtype': p = p.double()
    if bad == 'nan': p[0, 0, 0, 0] = float('nan')
    if bad == 'shape': p = p[:, :, :1]
    with pytest.raises(ValueError): api.validate_prototypes(p, c, n, (2, 2, 2), 3, 5)


def test_memory_fixed_round_weighted_batches_missing_classes_and_resume():
    api = _api()
    memory = api.GlobalPrototypeMemory(3, (1,))
    memory.start_round(1)
    memory.accumulate(torch.tensor([[1.], [7.]], requires_grad=True), torch.tensor([0, 1]), torch.tensor([2, 1]))
    memory.accumulate(torch.tensor([[5.]]), torch.tensor([0]), torch.tensor([6]))
    assert not memory.valid.any()
    restored = api.GlobalPrototypeMemory(3, (1,))
    restored.load_state_dict(memory.state_dict())
    stats = restored.finish_round()
    assert restored.values[:, 0].tolist() == [4., 7., 0.]
    assert stats['sample_counts_by_class'] == [8, 1, 0]
    assert stats['valid_class_mask'] == [True, True, False]
    assert not restored.values.requires_grad and restored.values.grad_fn is None
    with pytest.raises(RuntimeError): restored.finish_round()
    restored.start_round(2)
    before = restored.values.clone()
    restored.accumulate(torch.tensor([[12.]]), torch.tensor([0]), torch.tensor([3]))
    assert torch.equal(before, restored.values)
    restored.finish_round()
    assert restored.values[:, 0].tolist() == [12., 7., 0.]


@pytest.mark.parametrize('reduction,expected', [('mse', 5.), ('l2_squared', 10.)])
def test_alignment_valid_class_denominator_detaches_global_and_server(reduction, expected):
    api = _api()
    memory = api.GlobalPrototypeMemory(3, (2,))
    memory.start_round(1)
    memory.accumulate(torch.tensor([[1., 1.]]), torch.tensor([0]), torch.tensor([4]))
    memory.finish_round(); memory.start_round(2)
    p = torch.tensor([[2., 4.], [100., 100.]], requires_grad=True)
    model = nn.Linear(2, 3)
    align = memory.alignment_loss(p, torch.tensor([0, 2]), torch.tensor([2, 50]), reduction)
    assert align.item() == pytest.approx(expected)
    align.backward()
    assert torch.count_nonzero(p.grad[0]) == 2 and torch.count_nonzero(p.grad[1]) == 0
    assert memory.values.grad is None and all(v.grad is None for v in model.parameters())


@pytest.mark.parametrize('kind', ['SD_JOB', 'PROTO_JOB'])
def test_server_weighted_losses_fast_gradient_sum_and_one_adam_step(cpu_boundary, kind):
    api = _api()
    model = _top(); expected = copy.deepcopy(model)
    server = api.PrototypeServer(model, _config()); server.start_round(1)
    z = torch.randn(5, 2, 2, 2)
    y = torch.tensor([0, 2, 2, 2, 2])
    if kind == 'SD_JOB':
        x, labels, counts = z, y, None
    else:
        x, labels, counts = api.make_local_prototypes(z, y)
    leaf = x.detach().clone().requires_grad_()
    opt = torch.optim.Adam(expected.parameters(), lr=.001)
    if kind == 'SD_JOB':
        sd_logits = expected(leaf); sd_loss = F.cross_entropy(sd_logits, labels)
        p = torch.stack([leaf[y == c].mean(0) for c in (0, 2)])
        classes, n = torch.tensor([0, 2]), torch.tensor([1, 4])
    else:
        p, classes, n = leaf, labels, counts
        sd_loss = leaf.sum() * 0
    proto_logits = expected(p)
    proto_loss = (F.cross_entropy(proto_logits, classes, reduction='none') * n).sum() / n.sum()
    loss = .5 * (sd_loss + proto_loss) if kind == 'SD_JOB' else proto_loss
    loss.backward(); wanted_grad = leaf.grad.clone(); opt.step()
    result = server.process_batch([_packet(x, labels, kind, counts)])
    item = result['jobs'][0]
    assert result['samples'] == 5 and server.version == 1
    assert item['L_SD'] == pytest.approx(sd_loss.item())
    assert item['L_ProtoCE'] == pytest.approx(proto_loss.item())
    assert item['L_Align'] == 0 and item['L_Pseudo'] == 0
    assert item['L_total'] == pytest.approx(loss.item())
    assert torch.allclose(item['cut_gradient'], wanted_grad)
    assert all(torch.allclose(v, expected.state_dict()[k]) for k, v in model.state_dict().items())
    assert all(state['step'].item() == 1 for state in server.optimizer.state.values())
    assert item['server_forward_calls'] == (2 if kind == 'SD_JOB' else 1)
    assert item['proto_counts'] == [1, 4]


@pytest.mark.parametrize('kind,forwards', [('SD_JOB', 2), ('PROTO_JOB', 1)])
def test_bn_updates_per_forward_and_duplicate_job_rejected(cpu_boundary, kind, forwards):
    api = _api()
    server = api.PrototypeServer(_top(bn=True), _config()); server.start_round(1)
    x, labels = torch.randn(4, 2, 2, 2), torch.tensor([0, 0, 2, 2])
    counts = None
    if kind == 'PROTO_JOB': x, labels, counts = api.make_local_prototypes(x, labels)
    packet = _packet(x, labels, kind, counts)
    server.process_batch([packet])
    assert server.model[1].num_batches_tracked.item() == forwards
    with pytest.raises(ValueError): server.process_batch([packet])
    assert server.version == 1
    with pytest.raises(ValueError): server.process_batch([packet, packet])


def test_server_extra_state_roundtrip_restores_builder_versions_and_alignment(cpu_boundary):
    api = _api()
    server = api.PrototypeServer(_top(), _config())
    server.start_round(1)
    server.process_batch([_packet(torch.randn(4, 2, 2, 2), torch.tensor([0, 0, 2, 2]))])
    state = server.extra_state_dict()
    fresh = api.PrototypeServer(_top(), _config()); fresh.load_extra_state_dict(state)
    assert fresh.version == 1
    assert fresh.finish_round() == server.finish_round()
    fresh.start_round(2)
    result = fresh.process_batch([_packet(torch.ones(2, 2, 2, 2), torch.tensor([0, 2]),
        'PROTO_JOB', torch.tensor([16, 40]), job=(2, 0, 0))])
    assert result['jobs'][0]['L_Align'] > 0


def test_server_rejects_nonfinite_before_optimizer_or_memory_mutation(cpu_boundary):
    api = _api()
    server = api.PrototypeServer(_top(), _config()); server.start_round(1)
    x = torch.randn(2, 2, 2, 2); x[0, 0, 0, 0] = float('inf')
    with pytest.raises(ValueError): server.process_batch([_packet(x, torch.tensor([0, 2]))])
    assert server.version == 0 and not server.optimizer.state
    assert server.finish_round()['sample_counts_by_class'] == [0, 0, 0]


@pytest.mark.parametrize('kind', ['SD_JOB', 'PROTO_JOB'])
@pytest.mark.parametrize('beta,reduction', [(0., 'mse'), (.1, 'mse'), (.1, 'l2_squared')])
def test_nondefault_weights_and_prior_round_alignment_match_independent_reference(cpu_boundary, kind, beta, reduction):
    api = _api()
    config = _config(fast_sd_weight=.2, fast_proto_weight=.8, slow_proto_weight=1.7,
        alignment_weight_beta=beta, alignment_reduction=reduction)
    model = _top(); expected = copy.deepcopy(model)
    server = api.PrototypeServer(model, config)
    server.start_round(1)
    previous = torch.stack([torch.ones(2, 2, 2), torch.ones(2, 2, 2) * 3])
    server.memory.accumulate(previous, torch.tensor([0, 2]), torch.tensor([2, 4]))
    server.finish_round(); server.start_round(2)
    z, y = torch.randn(5, 2, 2, 2), torch.tensor([0, 0, 1, 2, 2])
    if kind == 'SD_JOB': x, labels, counts = z, y, None
    else: x, labels, counts = api.make_local_prototypes(z, y)
    leaf = x.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam(expected.parameters(), lr=.001)
    if kind == 'SD_JOB':
        sd_loss = F.cross_entropy(expected(leaf), labels)
        p = torch.stack([leaf[y == c].mean(0) for c in (0, 1, 2)])
        c, n = torch.tensor([0, 1, 2]), torch.tensor([2, 1, 2])
    else:
        p, c, n = leaf, labels, counts
        sd_loss = leaf.sum() * 0
    proto_ce = (F.cross_entropy(expected(p), c, reduction='none') * n).sum() / n.sum()
    distances = torch.stack([(p[0] - previous[0]).square().flatten(),
                             (p[2] - previous[1]).square().flatten()])
    distances = distances.mean(1) if reduction == 'mse' else distances.sum(1)
    align = (distances * torch.tensor([2, 2])).sum() / 4
    loss = (.2 * sd_loss + .8 * (proto_ce + beta * align) if kind == 'SD_JOB'
        else 1.7 * (proto_ce + beta * align))
    loss.backward(); wanted_grad = leaf.grad.clone(); optimizer.step()
    item = server.process_batch([_packet(x, labels, kind, counts, job=(2, 0, 0))])['jobs'][0]
    assert item['L_Align'] == pytest.approx(align.item())
    assert item['L_total'] == pytest.approx(loss.item())
    assert torch.allclose(item['cut_gradient'], wanted_grad, rtol=1e-5, atol=1e-7)
    assert all(torch.allclose(v, expected.state_dict()[k]) for k, v in model.state_dict().items())
    assert torch.equal(server.memory.values[[0, 2]], previous)


@pytest.mark.parametrize('corruption', ['logits', 'parameter_grad', 'cut_grad'])
def test_nonfinite_computation_fails_before_step(cpu_boundary, corruption):
    api = _api()
    model = _top()
    if corruption == 'logits':
        model.register_forward_hook(lambda module, inputs, output: output * float('nan'))
    elif corruption == 'parameter_grad':
        next(model.parameters()).register_hook(lambda gradient: gradient * float('nan'))
    else:
        class BadCut(nn.Module):
            def __init__(self, model):
                super().__init__(); self.model = model
            def forward(self, x):
                x.register_hook(lambda gradient: gradient * float('nan'))
                return self.model(x)
        model = BadCut(model)
    server = api.PrototypeServer(model, _config()); server.start_round(1)
    with pytest.raises(ValueError):
        server.process_batch([_packet(torch.randn(3, 2, 2, 2), torch.tensor([0, 2, 2]))])
    assert server.version == 0 and not server.optimizer.state
    assert server.finish_round()['sample_counts_by_class'] == [0, 0, 0]


@pytest.mark.parametrize('kind', ['SD_JOB', 'PROTO_JOB'])
def test_alignment_changes_client_gradient_but_not_direct_server_parameter_gradient(cpu_boundary, kind):
    api = _api()
    left = api.PrototypeServer(_top(), _config(alignment_weight_beta=0.))
    right = api.PrototypeServer(copy.deepcopy(left.model), _config(alignment_weight_beta=1.))
    for server in (left, right):
        server.start_round(1)
        server.memory.accumulate(torch.ones(2, 2, 2, 2), torch.tensor([0, 2]), torch.tensor([1, 4]))
        server.finish_round(); server.start_round(2)
    x, y = torch.randn(5, 2, 2, 2), torch.tensor([0, 2, 2, 2, 2])
    counts = None
    if kind == 'PROTO_JOB': x, y, counts = api.make_local_prototypes(x, y)
    packet = _packet(x, y, kind, counts, job=(2, 0, 0))
    a = left.process_batch([packet])['jobs'][0]
    b = right.process_batch([packet])['jobs'][0]
    assert not torch.allclose(a['cut_gradient'], b['cut_gradient'])
    assert all(torch.equal(value, right.model.state_dict()[key]) for key, value in left.model.state_dict().items())
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(left.model.parameters(), right.model.parameters()))
    assert left.memory.values.grad is None and right.memory.values.grad is None

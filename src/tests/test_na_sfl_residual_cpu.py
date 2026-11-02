"""V4 residual/pseudo mathematics on CPU; CUDA hardware boundaries only patched."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_na_sfl_prototype_cpu import _config, _packet, _top


def _residual():
    from utils import na_sfl_residual
    return na_sfl_residual


def _v4(**patch):
    return {'enabled': True, 'pseudo_loss_weight_gamma': .25,
        'pseudo_gradient_to_client': True, 'pseudo_samples_per_class': 4,
        'min_residuals_per_class': 2, 'max_residuals_per_class': 64,
        'residual_source': 'previous_completed_round', 'residual_recenter': True,
        'residual_scale': 1., 'residual_store_dtype': 'float32',
        'fallback_when_missing': 'proto_only', **patch}


def _bank(capacity=64, seed=42):
    return _residual().ResidualBank(3, (2, 2, 2), seed=seed, max_residuals_per_class=capacity)


def _add(bank, z, labels, job):
    from utils.na_sfl_prototype import make_local_prototypes
    p, c, n = make_local_prototypes(z, labels)
    bank.accumulate(z, labels, p, c, n, job)


def _values(entries):
    return {tuple(e['id']): e['residual'] for e in entries}


def test_exact_batch_local_detached_residuals_class_separation_and_singleton_skip():
    bank = _bank(); bank.start_round(1)
    z = torch.arange(5 * 8, dtype=torch.float32).reshape(5, 2, 2, 2).requires_grad_()
    labels = torch.tensor([2, 0, 2, 1, 2])
    _add(bank, z, labels, (1, 4, 0))
    assert all(not entries for entries in bank.bank)
    assert [len(e) for e in bank.builder] == [0, 0, 3]
    for entry in bank.builder[2]:
        index = entry['id'][-1]
        assert torch.equal(entry['residual'], z.detach()[index] - z.detach()[labels == 2].mean(0))
        assert entry['residual'].device.type == 'cpu'
        assert entry['residual'].dtype == torch.float32
        assert entry['residual'].grad_fn is None and not entry['residual'].requires_grad
    stats = bank.finish_round()
    assert stats['residual_bank_size_by_class'] == [0, 0, 3]
    assert stats['memory_bytes'] == 3 * 8 * 4
    assert stats['version'] == 1


def test_priority_reservoir_capacity_seed_order_independence_and_restore():
    torch.manual_seed(7)
    batches = [(torch.randn(12, 2, 2, 2), torch.tensor([0] * 6 + [2] * 6), (1, i, 0)) for i in range(4)]
    banks = [_bank(4, seed) for seed in (42, 42, 43)]
    before = torch.get_rng_state().clone()
    for index, bank in enumerate(banks):
        bank.start_round(1)
        for z, y, job in (reversed(batches) if index == 1 else batches): _add(bank, z, y, job)
    assert torch.equal(before, torch.get_rng_state())
    a, b, other = banks
    assert [e['id'] for e in a.builder[0]] == [e['id'] for e in b.builder[0]]
    assert [e['id'] for e in a.builder[0]] != [e['id'] for e in other.builder[0]]
    restored = _bank(4); restored.load_state_dict(a.state_dict())
    assert restored.finish_round() == a.finish_round()
    b.finish_round()
    for cls in (0, 2):
        assert len(a.bank[cls]) == 4
        assert all(torch.equal(value, _values(b.bank[cls])[key]) for key, value in _values(a.bank[cls]).items())
    restored.start_round(2)
    state = restored.state_dict(); state['bank'][0][0]['residual'].fill_(999)
    assert not torch.all(restored.bank[0][0]['residual'] == 999)


def test_first_round_and_missing_class_fallback_and_pseudo_mean():
    bank = _bank(); bank.start_round(1)
    p = torch.randn(2, 2, 2, 2, requires_grad=True)
    c, n = torch.tensor([0, 2]), torch.tensor([16, 40])
    empty = bank.make_pseudo(p, c, n, (1, 0, 0), _v4())
    assert empty['activation'] is None and empty['pseudo_samples_generated'] == 0
    assert empty['pseudo_fallback_classes'] == [0, 2]
    _add(bank, torch.arange(5 * 8, dtype=torch.float32).reshape(5, 2, 2, 2), torch.tensor([0] * 5), (1, 1, 0))
    # New candidates cannot affect the current immutable snapshot.
    assert bank.make_pseudo(p, c, n, (1, 0, 1), _v4())['activation'] is None
    bank.finish_round(); bank.start_round(2)
    before = torch.get_rng_state().clone()
    result = bank.make_pseudo(p, c, n, (2, 0, 0), _v4())
    assert torch.equal(before, torch.get_rng_state())
    assert result['pseudo_samples_generated'] == 4 and result['pseudo_fallback_classes'] == [2]
    assert torch.allclose(result['activation'].mean(0), p[0], atol=1e-5)
    assert result['labels'].tolist() == [0] * 4
    assert torch.allclose(result['weights'], torch.ones(4) / 4)
    result['activation'].sum().backward()
    assert torch.equal(p.grad[0], torch.ones_like(p[0]) * 4) and torch.count_nonzero(p.grad[1]) == 0
    again = bank.make_pseudo(p, c, n, (2, 0, 0), _v4())
    assert result['pseudo_donors'] == again['pseudo_donors']


@pytest.fixture
def cpu_boundary(monkeypatch):
    from utils import na_sfl_prototype as api, sflv2
    monkeypatch.setattr(sflv2, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(api, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(api, '_timed', lambda device, operation: (operation(), 0.))


def _server(beta=0., gamma=.25, to_client=True, bn=False):
    from utils.na_sfl_prototype import PrototypeServer
    config = _config(alignment_weight_beta=beta)
    config.update(method='na_sfl_v4_residual', seed=42,
        v4=_v4(pseudo_loss_weight_gamma=gamma, pseudo_gradient_to_client=to_client))
    return PrototypeServer(_top(bn=bn), config)


def _seed_donors(server):
    server.start_round(1)
    # Two donors for class0, four for class2: sample-mean CE would be incorrect.
    z = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 2, 2, 2) / 20
    _add(server.residual_bank, z, torch.tensor([0, 0, 2, 2, 2, 2]), (1, 1, 0))
    server.finish_round(); server.start_round(2)


@pytest.mark.parametrize('to_client', [True, False])
def test_pseudo_weighted_ce_and_client_toggle_match_independent_reference(cpu_boundary, to_client):
    server = _server(to_client=to_client); _seed_donors(server)
    expected = copy.deepcopy(server.model)
    p = torch.randn(2, 2, 2, 2)
    classes, counts = torch.tensor([0, 2]), torch.tensor([7, 3])
    leaf = p.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam(expected.parameters(), lr=.001)
    ce = (F.cross_entropy(expected(leaf), classes, reduction='none') * counts).sum() / counts.sum()
    generated = server.residual_bank.make_pseudo(leaf, classes, counts, (2, 0, 0), _v4(pseudo_gradient_to_client=to_client))
    logits = expected(generated['activation'])
    losses = F.cross_entropy(logits, generated['labels'], reduction='none')
    pseudo = .7 * losses[:2].mean() + .3 * losses[2:].mean()
    assert generated['labels'].tolist() == [0, 0, 2, 2, 2, 2]
    total = ce + .25 * pseudo
    total.backward(); wanted = leaf.grad.clone(); optimizer.step()
    item = server.process_batch([_packet(p, classes, 'PROTO_JOB', counts, (2, 0, 0))])['jobs'][0]
    assert item['L_Pseudo'] == pytest.approx(pseudo.item())
    assert item['L_total'] == pytest.approx(total.item())
    assert torch.allclose(item['cut_gradient'], wanted, rtol=1e-5, atol=1e-7)
    assert all(torch.allclose(v, expected.state_dict()[k]) for k, v in server.model.state_dict().items())
    assert all(state['step'].item() == 1 for state in server.optimizer.state.values())
    assert item['server_forward_calls'] == 2 and item['server_pseudo_forward_calls'] == 1
    assert item['pseudo_samples_generated'] == 6 and item['pseudo_fallback_classes'] == []


def test_gamma_zero_matches_v3_gradients_model_bn_and_rng(cpu_boundary):
    from utils.na_sfl_prototype import PrototypeServer
    v4 = _server(gamma=0., bn=True)
    v3 = PrototypeServer(copy.deepcopy(v4.model), _config(alignment_weight_beta=0.))
    for server in (v3, v4): server.start_round(1)
    z, labels = torch.randn(5, 2, 2, 2), torch.tensor([0, 0, 2, 2, 2])
    packets = [_packet(z, labels, job=(1, 1, 0)),
        _packet(z[[0, 2]], torch.tensor([0, 2]), 'PROTO_JOB', torch.tensor([2, 3]), (1, 0, 0))]
    for packet in packets:
        before = torch.get_rng_state().clone()
        a = v3.process_batch([packet])['jobs'][0]; b = v4.process_batch([packet])['jobs'][0]
        assert torch.equal(before, torch.get_rng_state())
        assert torch.equal(a['cut_gradient'], b['cut_gradient'])
        assert b['server_pseudo_forward_calls'] == 0 and b['pseudo_samples_generated'] == 0
        assert all(torch.equal(v, v4.model.state_dict()[k]) for k, v in v3.model.state_dict().items())
    for server in (v3, v4): server.finish_round(); server.start_round(2)
    packet = _packet(z[[0, 2]], torch.tensor([0, 2]), 'PROTO_JOB', torch.tensor([2, 3]), (2, 0, 0))
    a = v3.process_batch([packet])['jobs'][0]; b = v4.process_batch([packet])['jobs'][0]
    assert torch.equal(a['cut_gradient'], b['cut_gradient']) and b['server_forward_calls'] == 1
    assert all(torch.equal(v, v4.model.state_dict()[k]) for k, v in v3.model.state_dict().items())


def test_v4_checkpoint_restores_immutable_bank_current_builder_and_selection(cpu_boundary):
    server = _server(); _seed_donors(server)
    z = torch.randn(4, 2, 2, 2)
    server.process_batch([_packet(z, torch.tensor([0, 0, 2, 2]), job=(2, 1, 0))])
    restored = _server(); restored.load_extra_state_dict(server.extra_state_dict())
    p, c, n = z[[0, 2]], torch.tensor([0, 2]), torch.tensor([3, 7])
    a = server.residual_bank.make_pseudo(p, c, n, (2, 0, 0), _v4())
    b = restored.residual_bank.make_pseudo(p, c, n, (2, 0, 0), _v4())
    assert torch.equal(a['activation'], b['activation']) and a['pseudo_donors'] == b['pseudo_donors']
    assert server.finish_round() == restored.finish_round()


@pytest.mark.parametrize('corrupt', ['shape', 'dtype', 'nan', 'id', 'class', 'capacity',
    'rng', 'builder_without_job', 'inactive_source'])
def test_restore_rejects_invalid_features_identity_capacity_rng_and_lifecycle(corrupt):
    bank = _bank(4); bank.start_round(1)
    z = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 2, 2, 2)
    _add(bank, z, torch.tensor([0, 0, 2, 2, 2, 2]), (1, 1, 0))
    if corrupt == 'builder_without_job':
        state = bank.state_dict(); state['seen_jobs'] = []
    else:
        bank.finish_round(); state = bank.state_dict()
        entry = state['bank'][0][0]
        if corrupt == 'shape': entry['residual'] = entry['residual'][:1]
        if corrupt == 'dtype': entry['residual'] = entry['residual'].double()
        if corrupt == 'nan': entry['residual'][0, 0, 0] = float('nan')
        if corrupt == 'id': entry['id'][0] = 0
        if corrupt == 'class': state['bank'][1], state['bank'][0] = state['bank'][0], []
        if corrupt == 'capacity': state['bank'][0] *= 3
        if corrupt == 'rng': state['selection_rng_state']['seed'] = 99
        if corrupt == 'inactive_source':
            state['round_id'] = 2; state['seen_jobs'] = []
    with pytest.raises(ValueError): _bank(4).load_state_dict(state)


def test_positive_gamma_uses_extra_bn_forward_only_when_same_class_donors_exist(cpu_boundary):
    server = _server(bn=True)
    server.start_round(1)
    p, c, n = torch.randn(2, 2, 2, 2), torch.tensor([0, 2]), torch.tensor([16, 40])
    item = server.process_batch([_packet(p, c, 'PROTO_JOB', n, (1, 0, 0))])['jobs'][0]
    assert item['server_pseudo_forward_calls'] == 0 and item['pseudo_fallback_classes'] == [0, 2]
    assert server.model[1].num_batches_tracked.item() == 1
    _add(server.residual_bank, torch.randn(4, 2, 2, 2), torch.tensor([0, 0, 2, 2]), (1, 1, 0))
    server.finish_round(); server.start_round(2)
    item = server.process_batch([_packet(p, c, 'PROTO_JOB', n, (2, 0, 0))])['jobs'][0]
    assert item['server_pseudo_forward_calls'] == 1 and item['server_forward_calls'] == 2
    assert server.model[1].num_batches_tracked.item() == 3
    assert all(state['step'].item() == 2 for state in server.optimizer.state.values())


def test_pseudo_client_gradient_toggle_changes_only_input_gradient(cpu_boundary):
    yes, no = _server(to_client=True), _server(to_client=False)
    for server in (yes, no): _seed_donors(server)
    p, c, n = torch.randn(2, 2, 2, 2), torch.tensor([0, 2]), torch.tensor([7, 3])
    packet = _packet(p, c, 'PROTO_JOB', n, (2, 0, 0))
    a, b = yes.process_batch([packet])['jobs'][0], no.process_batch([packet])['jobs'][0]
    assert not torch.allclose(a['cut_gradient'], b['cut_gradient'])
    assert all(torch.equal(v, no.model.state_dict()[k]) for k, v in yes.model.state_dict().items())
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(yes.model.parameters(), no.model.parameters()))


def test_default_cut2_residual_capacity_is_eighty_mib_per_snapshot_and_builder():
    bank = _residual().ResidualBank(10, (128, 16, 16))
    assert bank.capacity_memory_bytes == 80 * 1024 ** 2
    assert bank.stats()['memory_bytes'] == 0 and bank.stats()['builder_memory_bytes'] == 0

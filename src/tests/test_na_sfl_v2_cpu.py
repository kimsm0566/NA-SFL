"""Persistent CPU contracts for no-wait Concat and phase-local pacing."""
import copy
from collections import deque
import threading

import pytest
import torch
from torch import nn
from torch.nn import functional as F


def test_v2_config_and_four_way_comparison_share_network_and_client_modes():
    from utils.na_sfl_config import load_config, comparison_configs
    cfg = load_config(overrides={'method': 'na_sfl_v2', 'server_policy': 'concat',
        'aux_client_count': 2, 'network': {'slow_client_ids': [0, 5]}})
    rows = comparison_configs(cfg)
    assert [(r['method'], r['server_policy'], r['aux_client_count']) for r in rows] == [
        ('splitfedv2', 'individual', 0), ('splitfedv2', 'concat', 0),
        ('na_sfl_v1', 'individual', 2), ('na_sfl_v2', 'concat', 2)]
    assert all(r['network'] == rows[0]['network'] for r in rows)
    assert all(r['server_max_batch_samples'] == 512 for r in rows)
    assert set(rows[-1]['aux_client_ids']) == {0, 5} and rows[-1]['auxiliary_fedavg']
    for patch in ({'server_max_batch_samples': 127}, {'server_policy': 'wait_for_full'},
                  {'method': 'na_sfl_v2', 'server_policy': 'individual'},
                  {'network': {'slow_client_ids': [0, 0]}},
                  {'network': {'slow_uplink_mbps': 0}},
                  {'client_compute_seconds': {'forward': -1}}):
        with pytest.raises(ValueError):
            load_config(overrides=patch)


def test_greedy_batch_takes_only_fifo_prefix_without_splitting_or_waiting():
    from utils.na_sfl_batch import take_ready_batch
    def packet(n, name):
        return {'row': {'samples': n, 'job_id': name}}
    queue = deque([packet(128, 'a'), packet(128, 'b'), packet(56, 'c'), packet(128, 'd')])
    assert [p['row']['job_id'] for p in take_ready_batch(queue, 312, 'concat')] == ['a', 'b', 'c']
    assert [p['row']['job_id'] for p in queue] == ['d']
    assert len(take_ready_batch(queue, 512, 'individual')) == 1
    assert take_ready_batch(queue, 512, 'concat') == []
    with pytest.raises(ValueError):
        take_ready_batch(deque([packet(128, 'big')]), 127, 'concat')


def _top():
    torch.manual_seed(18)
    return nn.Sequential(nn.Conv2d(2, 3, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())


@pytest.fixture
def cpu_cuda_boundary(monkeypatch):
    # Replace only the hardware boundary; autograd/loss/Adam stay real.
    from utils import sflv2
    monkeypatch.setattr(sflv2, '_model_device', lambda model: torch.device('cpu'))
    monkeypatch.setattr(sflv2, '_timed', lambda device, operation: (operation(), 0.))


def test_mixed_concat_uses_one_sample_mean_adam_step_and_local_mean_cut_gradients(cpu_cuda_boundary):
    from utils.sflv2 import SFLV2Server
    model = _top()
    expected = copy.deepcopy(model)
    config = {'optimizer': 'adam', 'lr': .001, 'weight_decay': 0., 'num_classes': 3}
    server = SFLV2Server(model, config)
    acts = [torch.randn(n, 2, 2, 2) for n in (3, 2, 1)]
    labels = [torch.arange(len(x)) % 3 for x in acts]
    leaf = torch.cat(acts).requires_grad_()
    opt = torch.optim.Adam(expected.parameters(), lr=.001)
    losses = F.cross_entropy(expected(leaf), torch.cat(labels), reduction='none')
    losses.mean().backward()
    wanted = leaf.grad.detach().clone()
    opt.step()
    result = server.process_batch([
        {'job_id': (1, i, 0), 'activation': x, 'labels': y, 'return_cut_gradient': i != 1}
        for i, (x, y) in enumerate(zip(acts, labels))])
    assert server.version == 1
    assert result['samples'] == 6 and len(result['jobs']) == 3
    assert all(torch.equal(v, expected.state_dict()[k]) for k, v in model.state_dict().items())
    offset = 0
    for i, row in enumerate(result['jobs']):
        n = len(acts[i])
        assert row['version_before'] == 0 and row['version_after'] == 1
        assert row['loss'] == pytest.approx(losses[offset:offset+n].mean().item())
        if i == 1:
            assert row['cut_gradient'] is None
        else:
            assert torch.allclose(row['cut_gradient'], wanted[offset:offset+n] * (6 / n))
        offset += n


def test_single_packet_batch_preserves_original_server_numerics_and_aux_only_has_no_gradient(cpu_cuda_boundary):
    from utils.sflv2 import SFLV2Server
    config = {'optimizer': 'adam', 'lr': .001, 'weight_decay': 0., 'num_classes': 3}
    a = SFLV2Server(_top(), config)
    b = SFLV2Server(copy.deepcopy(a.model), config)
    x, y = torch.randn(2, 2, 2, 2), torch.tensor([0, 2])
    old = a.process((1, 0, 0), x, y)
    new = b.process_batch([{'job_id': (1, 0, 0), 'activation': x, 'labels': y,
                           'return_cut_gradient': True}])['jobs'][0]
    assert torch.equal(old['cut_gradient'], new['cut_gradient'])
    assert all(torch.equal(v, b.model.state_dict()[k]) for k, v in a.model.state_dict().items())
    result = b.process_batch([{'job_id': (2, 0, 0), 'activation': x, 'labels': y,
                               'return_cut_gradient': False}] * 2)
    assert all(row['cut_gradient'] is None for row in result['jobs'])
    assert b.version == 2


def test_phase_padding_subtracts_elapsed_and_records_overrun_without_device_barrier():
    from utils.na_sfl_batch import pace_phase, phase_targets
    class Clock:
        value = .04
        def __call__(self): return self.value
    class Event:
        def wait(self, seconds): clock.value += seconds; return False
        def is_set(self): return False
    clock = Clock()
    result = pace_phase(0., .1, Event(), clock=clock)
    assert result['padding_s'] == pytest.approx(.06)
    assert result['overrun_s'] == 0
    assert clock.value == pytest.approx(.1)
    result = pace_phase(0., .01, Event(), clock=clock)
    assert result['padding_s'] == 0 and result['overrun_s'] == pytest.approx(.09)
    config = {'batch_size': 128, 'client_compute_seconds': {'forward': .3, 'backward': .6, 'auxiliary': .1}}
    assert phase_targets(config, 128, 'aux')['auxiliary'] == pytest.approx(.7)
    assert phase_targets(config, 56, 'server')['backward'] == pytest.approx(.6 * 56 / 128)
    aborted = threading.Event(); aborted.set()
    with pytest.raises(RuntimeError):
        pace_phase(0., 1., aborted, clock=clock)


def test_grouped_resume_validates_server_steps_separately_from_sd_and_membership():
    from utils.na_sfl_train import validate_resume_metadata
    from utils.deit_async import async_trace_sha256
    jobs = [{'round': 1, 'job_id': (1, i, 0), 'samples': 2, 'client_id': i,
             'arrival_seq': i, 'server_batch_id': 0, 'server_version_before': 0,
             'server_version_after': 1, 'client_version_before': 0, 'client_version_after': 1}
            for i in range(2)]
    batches = [{'round': 1, 'batch_id': 0, 'server_version_before': 0,
                'server_version_after': 1, 'samples': 4, 'sd_count': 2,
                'job_ids': [job['job_id'] for job in jobs]}]
    boundary = {'round': 1, 'success': True, 'queue_empty': True, 'streams_completed': True,
        'unresolved_futures': 0, 'workers_alive': 0, 'pending_graphs': 0,
        'trace_sha256': async_trace_sha256(jobs),
        'server_batch_trace_sha256': async_trace_sha256(batches),
        'runtimecounts': {'admitted_jobs': 2, 'serviced_jobs': 2, 'completed_jobs': 2, 'serviced_batches': 1}}
    saved = {'next_round': 1, 'config': {'n_clients': 2, 'server_policy': 'concat',
        'server_max_batch_samples': 4}, 'rounds': [{'round': 1}], 'job_records': jobs,
        'server_batch_records': batches, 'execution_boundaries': [boundary],
        'aggregation_records': [{'round': 1}], 'server_version': 1, 'client_versions': [1, 1],
        'client_optimizers': [{'state': {}}, {'state': {}}], 'auxiliary_optimizers': []}
    validate_resume_metadata(saved)
    for kind in ('membership', 'step', 'capacity', 'transport'):
        bad = copy.deepcopy(saved)
        if kind == 'membership': bad['server_batch_records'][0]['job_ids'].pop()
        if kind == 'step': bad['server_version'] = 2
        if kind == 'capacity': bad['config']['server_max_batch_samples'] = 3
        if kind == 'transport':
            bad['execution_boundaries'][0]['transport'] = {
                'workers_alive': 0, 'pending_sd': 1, 'pending_gradients': 0}
        with pytest.raises(ValueError):
            validate_resume_metadata(bad)


def test_four_condition_cli_uses_existing_entrypoint_and_rejects_two_dispatchers():
    from utils.option import parse_args
    args = parse_args(['--comparison', '--config', 'configs/na_sfl_network.yaml'])
    assert args.comparison and args.config['network']['slow_client_ids'] == [0, 5]
    assert sum(args.config['client_compute_seconds'].values()) == pytest.approx(1.)
    with pytest.raises(SystemExit):
        parse_args(['--comparison', '--sweep'])


def test_network_fedavg_uses_received_uploads_and_received_broadcasts():
    from types import SimpleNamespace
    from utils.na_sfl_train import NASFLTrainer
    owner = NASFLTrainer.__new__(NASFLTrainer)
    owner.args = {'auxiliary_fedavg': True}
    owner.client_optimizer_config = owner.aux_optimizer_config = {'optimizer': 'adam', 'lr': .001}
    owner.server = SimpleNamespace(version=3)
    owner.aggregation_records = []
    owner.clients = [SimpleNamespace(model=nn.Linear(1, 1, bias=False), auxiliary_head=nn.Linear(1, 1),
        pending=None) for _ in range(2)]
    class Network:
        calls = []
        def transfer_models(self, states, direction, tag):
            self.calls.append(direction)
            states = [{key: value.clone() for key, value in state.items()} for state in states]
            for i, state in enumerate(states):
                if direction == 'upload': state['client:weight'].fill_(10. * (i + 1))
                else: state['client:weight'].add_(1.)
            return states
    owner.network = Network()
    owner._aggregate([1, 1], 1)
    assert owner.network.calls == ['upload', 'download']
    assert all(node.model.weight.item() == 16. for node in owner.clients)
    assert owner.server.version == 3


def test_tta_reports_actual_elapsed_at_first_evaluated_target_crossing():
    from utils.na_sfl_artifacts import target_summary
    rows = [{'round': 1, 'test_accuracy': 65., 'elapsed_wall_s': 90.},
            {'round': 2, 'test_accuracy': 71., 'elapsed_wall_s': 160.}]
    assert target_summary(rows, 70.)['elapsed_wall_s'] == 160.
    assert target_summary(rows, 80.)['elapsed_wall_s'] is None


def test_tta_counts_initial_target_even_if_accuracy_drops_after_training():
    from utils.na_sfl_artifacts import target_summary
    result = target_summary([{'round': 1, 'test_accuracy': 8., 'elapsed_wall_s': 80.}], 9.,
                            initial_evaluation={'test_accuracy': 10., 'elapsed_wall_s': 4.})
    assert result == {'accuracy': 9., 'status': 'reached', 'round': 0, 'elapsed_wall_s': 4.}

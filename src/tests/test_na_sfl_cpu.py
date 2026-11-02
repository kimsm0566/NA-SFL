"""CPU-only contracts for the new NA path; never invokes a CUDA suite."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from data.na_sfl_data import balanced_partitions
from utils.na_sfl_config import load_config, select_aux_clients
from utils.na_sfl import auxiliary_update, average_round_models
from utils.na_sfl_artifacts import retain_final_checkpoint, target_summary, discard_failed_checkpoint, sweep_status
from utils.na_sfl_train import validate_resume_metadata


@pytest.mark.parametrize('samples,alpha', [(3000, .1), (5000, .1), (5000, .5)])
def test_balanced_dirichlet_has_exact_disjoint_quotas(samples, alpha):
    labels = np.repeat(np.arange(10), 5000)
    parts = balanced_partitions(labels, 10, samples, alpha, 42)
    assert [len(p) for p in parts] == [samples] * 10
    flat = np.concatenate(parts)
    assert len(np.unique(flat)) == 10 * samples
    assert flat.min() >= 0 and flat.max() < 50000
    if samples == 5000:
        assert np.array_equal(np.sort(flat), np.arange(50000))
    histogram = np.stack([np.bincount(labels[p], minlength=10) for p in parts])
    assert np.all(histogram.sum(axis=1) == samples)
    assert histogram.max() - histogram.min() > 500
    repeated = balanced_partitions(labels, 10, samples, alpha, 42)
    assert all(np.array_equal(a, b) for a, b in zip(parts, repeated))


def test_balanced_iid_and_invalid_budgets():
    labels = np.repeat(np.arange(10), 50)
    parts = balanced_partitions(labels, 10, 30, .1, 7, partition='iid')
    assert len(np.unique(np.concatenate(parts))) == 300
    assert all(len(p) == 30 for p in parts)
    for budget in (0, 51):
        with pytest.raises(ValueError):
            balanced_partitions(labels, 10, budget, .1, 7)
    with pytest.raises(ValueError):
        balanced_partitions(labels, 10, 30, 0., 7)


def test_mode_selection_is_fixed_nested_and_does_not_consume_global_rng():
    state = np.random.get_state()
    selections = [select_aux_clients(42, 10, k) for k in range(11)]
    assert selections[0] == []
    assert sorted(selections[-1]) == list(range(10))
    assert all(len(ids) == k for k, ids in enumerate(selections))
    assert all(set(a) < set(b) for a, b in zip(selections, selections[1:]))
    assert np.array_equal(state[1], np.random.get_state()[1])


def test_config_uses_requested_live_adam_and_rejects_network_or_invalid_modes():
    cfg = load_config(overrides={'samples_per_client': 3000, 'dirichlet_alpha': .1,
                                 'optimizer': 'adam', 'lr': .001, 'aux_client_count': 7})
    assert cfg['training_subset'] == 30000
    assert len(cfg['aux_client_ids']) == 7
    assert cfg['mode'] == 'live_async' and cfg['num_classes'] == 10
    for patch in ({'mode': 'sync'}, {'aux_client_count': 11}, {'aux_client_count': True},
                  {'uplink_mbps': 100}, {'lr': float('nan')},
                  {'method': 'splitfedv2', 'aux_client_count': 1}):
        with pytest.raises(ValueError):
            load_config(overrides=patch)


def _toy():
    torch.manual_seed(123)
    client = nn.Linear(2, 3)
    head = nn.Linear(3, 2)
    client_opt = torch.optim.Adam(client.parameters(), lr=.001)
    head_opt = torch.optim.Adam(head.parameters(), lr=.001)
    return client, head, client_opt, head_opt


def test_server_mode_auxiliary_changes_only_head_and_preserves_client_graph():
    client, head, client_opt, head_opt = _toy()
    before_client, before_head = copy.deepcopy(client.state_dict()), copy.deepcopy(head.state_dict())
    activation = client(torch.tensor([[1., 2.], [3., -1.]]))
    result = auxiliary_update(activation, torch.tensor([0, 1]), head, head_opt)
    assert result['samples'] == 2 and result['loss'] > 0
    assert all(torch.equal(p, before_client[k]) for k, p in client.state_dict().items())
    assert all(p.grad is None for p in client.parameters())
    assert any(not torch.equal(p, before_head[k]) for k, p in head.state_dict().items())
    activation.backward(torch.ones_like(activation))
    client_opt.step()
    assert any(not torch.equal(p, before_client[k]) for k, p in client.state_dict().items())


def test_aux_mode_updates_client_without_server_and_preserves_sd_snapshot():
    client, head, client_opt, head_opt = _toy()
    before_client, before_head = copy.deepcopy(client.state_dict()), copy.deepcopy(head.state_dict())
    activation = client(torch.tensor([[1., 2.], [3., -1.]]))
    snapshot = activation.detach().clone()
    expected_snapshot = snapshot.clone()
    auxiliary_update(activation, torch.tensor([0, 1]), head, head_opt, client_opt)
    assert any(not torch.equal(p, before_client[k]) for k, p in client.state_dict().items())
    assert any(not torch.equal(p, before_head[k]) for k, p in head.state_dict().items())
    assert torch.equal(snapshot, expected_snapshot) and not snapshot.requires_grad


def test_auxiliary_round_average_preserves_server_and_uses_max_integer_bn_buffer():
    clients = [nn.BatchNorm1d(2), nn.BatchNorm1d(2)]
    heads = [nn.BatchNorm1d(2), nn.BatchNorm1d(2)]
    server = nn.Linear(2, 2)
    before_server = copy.deepcopy(server.state_dict())
    with torch.no_grad():
        clients[0].weight.fill_(2.); clients[1].weight.fill_(6.)
        heads[0].running_mean.fill_(10.); heads[1].running_mean.fill_(14.)
        heads[0].num_batches_tracked.fill_(3); heads[1].num_batches_tracked.fill_(7)
    average_round_models(clients, heads, [3000, 3000], auxiliary_fedavg=True)
    assert all(torch.equal(m.weight, torch.tensor([4., 4.])) for m in clients)
    assert all(torch.equal(m.running_mean, torch.tensor([12., 12.])) for m in heads)
    assert all(m.num_batches_tracked.item() == 7 for m in heads)
    assert all(torch.equal(value, before_server[key]) for key, value in server.state_dict().items())


def test_no_aux_average_retains_distinct_local_heads():
    clients = [nn.Linear(2, 2), nn.Linear(2, 2)]
    heads = [nn.Linear(2, 2), nn.Linear(2, 2)]
    before = [copy.deepcopy(head.state_dict()) for head in heads]
    average_round_models(clients, heads, [1, 1], auxiliary_fedavg=False)
    assert all(torch.equal(value, before[i][key])
               for i, head in enumerate(heads) for key, value in head.state_dict().items())


def test_round_averaging_resets_optimizers_without_copying_models_for_hashes(monkeypatch):
    from utils import na_sfl_train
    owner = na_sfl_train.NASFLTrainer.__new__(na_sfl_train.NASFLTrainer)
    owner.args = {'auxiliary_fedavg': True}
    owner.client_optimizer_config = owner.aux_optimizer_config = {
        'optimizer': 'adam', 'lr': .001, 'weight_decay': 0.}
    owner.clients = []
    for client_value, head_value in ((2., 10.), (6., 14.)):
        model, head = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        opt, aux_opt = torch.optim.Adam(model.parameters()), torch.optim.Adam(head.parameters())
        for network, optimizer, value in ((model, opt, client_value), (head, aux_opt, head_value)):
            network.weight.grad = torch.ones_like(network.weight)
            optimizer.step()
            with torch.no_grad():
                network.weight.fill_(value)
        owner.clients.append(SimpleNamespace(model=model, auxiliary_head=head, pending=None,
            optimizer=opt, auxiliary_optimizer=aux_opt))
    server = nn.Linear(1, 1, bias=False)
    server_before = copy.deepcopy(server.state_dict())
    owner.server = SimpleNamespace(model=server, version=17)
    owner.aggregation_records = []
    def forbidden_hash(_model):
        raise AssertionError('round averaging must not copy model weights to CPU for hashing')
    monkeypatch.setattr(na_sfl_train, 'model_digest', forbidden_hash)
    owner._aggregate([3000, 3000], 1)
    assert all(node.model.weight.item() == 4. for node in owner.clients)
    assert all(node.auxiliary_head.weight.item() == 12. for node in owner.clients)
    assert all(not node.optimizer.state and not node.auxiliary_optimizer.state for node in owner.clients)
    assert all(torch.equal(value, server_before[key]) for key, value in server.state_dict().items())
    assert owner.server.version == 17


def test_na_cli_keeps_single_entrypoint_and_normalizes_count():
    from utils.option import parse_args
    args = parse_args(['na-sfl-v1', '--aux-client-count', '10', '--seed', '42', '--rounds', '2'])
    assert args.config['aux_client_count'] == 10
    assert len(args.config['aux_client_ids']) == 10
    assert args.config['samples_per_client'] == 3000 and args.config['rounds'] == 2


@pytest.mark.parametrize('name', ['NVIDIA RTX A5000', 'Quadro RTX 5000'])
def test_requested_training_gpus_are_accepted_but_busy_devices_are_rejected(name):
    from utils.na_sfl_run import validate_training_gpu
    gpu = {'name': name, 'compute_processes': [], 'free_bytes': 90, 'total_bytes': 100}
    validate_training_gpu(gpu)
    for patch in ({'compute_processes': ['another training process']}, {'free_bytes': 70}):
        with pytest.raises(RuntimeError):
            validate_training_gpu({**gpu, **patch})


def test_cli_defaults_to_na_and_rejects_removed_commands():
    from utils.option import parse_args
    args = parse_args([])
    assert args.trainer == 'na-sfl-v1'
    assert args.config['mode'] == 'live_async'
    assert parse_args(['--aux-client-count', '5']).config['aux_client_count'] == 5
    for removed in ('fedavg', 'microsfl', 'microsfl-threaded', 'sflv2-baseline', 'sflv2-sync'):
        with pytest.raises(SystemExit) as error:
            parse_args([removed])
        assert error.value.code == 2


def test_target_round_reports_unreached_without_invented_round():
    rows = [{'round': 1, 'test_accuracy': 20.}, {'round': 2, 'test_accuracy': 35.}]
    assert target_summary(rows, 30.) == {'accuracy': 30., 'status': 'reached', 'round': 2}
    assert target_summary(rows, 80.) == {'accuracy': 80., 'status': 'not_reached', 'round': None}
    assert target_summary(rows, None)['status'] == 'not_configured'


def _entry(root, number, experiment=None):
    path = root / ('run' + str(number)) / 'final.pt'
    path.parent.mkdir(parents=True)
    path.write_bytes(('synthetic retention fixture ' + str(number)).encode())
    return {'experiment_id': experiment or str(number), 'path': str(path),
            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'final_saved_at': f'2026-10-30T12:00:{number:02d}+09:00'}


def _write_registry_in_process(root, entry, pipe):
    try:
        pipe.send('started')
        retain_final_checkpoint(root / 'registry.json', entry, root)
        pipe.send('finished')
    finally:
        pipe.close()


def test_checkpoint_registry_waits_for_other_process_before_updating(tmp_path):
    import fcntl
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    entry = _entry(tmp_path, 1)
    process = context.Process(target=_write_registry_in_process, args=(tmp_path, entry, send))
    with (tmp_path / 'registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        process.start()
        send.close()
        try:
            assert receive.poll(15) and receive.recv() == 'started'
            assert not receive.poll(.3), 'registry write bypassed the other process lock'
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            process.join(15)
            if process.is_alive():
                process.terminate()
                process.join()
    assert process.exitcode == 0
    assert receive.poll(1) and receive.recv() == 'finished'
    assert json.loads((tmp_path / 'registry.json').read_text())['final_count'] == 1


def test_checkpoint_retention_keeps_latest_five_and_replaces_same_experiment(tmp_path):
    registry = tmp_path / 'registry.json'
    entries = [_entry(tmp_path, i) for i in range(6)]
    registry.write_text(json.dumps({'final_checkpoints': entries[:5], 'final_count': 5, 'global_limit': 5}))
    saved = retain_final_checkpoint(registry, entries[5], tmp_path)
    assert saved['final_count'] == 5
    assert not (tmp_path / 'run0' / 'final.pt').exists()
    replacement = _entry(tmp_path, 6, experiment='5')
    saved = retain_final_checkpoint(registry, replacement, tmp_path)
    assert saved['final_count'] == 5
    assert not (tmp_path / 'run5' / 'final.pt').exists()
    assert (tmp_path / 'run6' / 'final.pt').exists()


def test_retention_refuses_paths_outside_results_root_before_deleting(tmp_path):
    allowed = tmp_path / 'results'
    allowed.mkdir()
    outside = _entry(tmp_path, 1)
    registry = allowed / 'registry.json'
    entry = _entry(allowed, 2)
    registry.write_text(json.dumps({'final_checkpoints': [outside]}))
    with pytest.raises(ValueError):
        retain_final_checkpoint(registry, entry, allowed)
    assert (tmp_path / 'run1' / 'final.pt').exists()


@pytest.mark.parametrize('registered', [False, True])
def test_failed_launch_removes_its_final_without_deleting_other_models(tmp_path, registered):
    run = tmp_path / 'failed'
    checkpoint = run / 'checkpoints' / 'final.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'failed-stage fixture')
    prior = _entry(tmp_path, 0)
    entries = [prior]
    if registered:
        entries.append({'experiment_id': 'failed', 'path': str(checkpoint)})
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'final_checkpoints': entries, 'final_count': len(entries)}))
    discard_failed_checkpoint(run, registry, tmp_path)
    assert not checkpoint.exists()
    assert (tmp_path / 'run0' / 'final.pt').exists()
    saved = json.loads(registry.read_text())
    assert saved['final_count'] == 1 and saved['final_checkpoints'][0] == prior


def test_short_sweep_does_not_claim_target_rounds_completed():
    assert sweep_status([{'completed': False}]) == {
        'status': 'stopped_at_round_boundary', 'completed': False, 'completed_dispatch': True}
    assert sweep_status([{'completed': True}, {'completed': True}])['status'] == 'completed'


def test_checkpoint_rng_touches_only_the_selected_cuda_device(monkeypatch):
    import random
    from utils.deit_artifacts import capture_rng, restore_rng
    selected = torch.tensor([12, 34], dtype=torch.uint8)
    restored = []
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda device: selected.clone() if device == 1 else None)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda state, device: restored.append((device, state)))
    def forbidden(*args):
        raise AssertionError('checkpoint must not initialize or change the other GPU')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', forbidden)
    scheduler = random.Random(42)
    saved = capture_rng(scheduler)
    assert saved['torch_cuda']['device'] == 1
    restore_rng(saved, scheduler)
    assert len(restored) == 1 and restored[0][0] == 1 and torch.equal(restored[0][1], selected)


def test_resume_rejects_changed_trace_or_undrained_boundary_or_momentum():
    from utils.deit_async import async_trace_sha256
    jobs = [{'round': 1, 'client_id': 0, 'arrival_seq': 0, 'server_version_before': 0,
             'server_version_after': 1, 'client_version_before': 0, 'client_version_after': 1}]
    boundary = {'round': 1, 'success': True, 'queue_empty': True, 'streams_completed': True,
        'unresolved_futures': 0, 'workers_alive': 0, 'pending_graphs': 0,
        'trace_sha256': async_trace_sha256(jobs),
        'runtimecounts': {'admitted_jobs': 1, 'serviced_jobs': 1, 'completed_jobs': 1}}
    saved = {'next_round': 1, 'config': {'n_clients': 1}, 'rounds': [{'round': 1}],
        'job_records': jobs, 'execution_boundaries': [boundary], 'aggregation_records': [{'round': 1}],
        'server_version': 1, 'client_versions': [1], 'client_optimizers': [{'state': {}}],
        'auxiliary_optimizers': [{'state': {}}]}
    validate_resume_metadata(saved)
    for key in ('trace', 'drain', 'optimizer', 'version'):
        bad = copy.deepcopy(saved)
        if key == 'trace': bad['job_records'][0]['client_version_after'] = 7
        if key == 'drain': bad['execution_boundaries'][0]['queue_empty'] = False
        if key == 'optimizer': bad['client_optimizers'][0]['state'] = {0: {}}
        if key == 'version': bad['server_version'] = 2
        with pytest.raises(ValueError):
            validate_resume_metadata(bad)

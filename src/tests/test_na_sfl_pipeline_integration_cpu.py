"""Configuration and persisted group invariants for authorized C-only mode."""
import copy

import pytest


METHOD = 'na_sfl_trainer_sender_pipeline'


def config(**extra):
    from utils.na_sfl_config import load_config
    return load_config(overrides={'method': METHOD, **extra})


def test_c_defaults_keep_original_v5_training_semantics():
    from utils.na_sfl_config import load_config
    from utils.option import parse_args
    c = config()
    assert c['bn_policy'] == 'dual_running_stats_shared_affine'
    assert c['random_sd'] == {'sample_ratio': .1, 'sample_seed': 42}
    assert c['pipeline']['max_inflight_per_client'] == c['pipeline']['accumulation_batches'] == 2
    assert c['pipeline']['per_client_send_worker'] == 1
    assert c['prototype']['fast_sd_weight'] == .8 and c['prototype']['fast_proto_weight'] == .2
    assert parse_args(['--method', METHOD]).config['method'] == METHOD
    old = load_config(overrides={'method': 'na_sfl_proto_random_sd'})
    assert old['pipeline'] == {} and old['bn_policy'] == c['bn_policy']


@pytest.mark.parametrize('changes', [
    {'pipeline': {'max_inflight_per_client': 1}},
    {'pipeline': {'accumulation_batches': 1}},
    {'pipeline': {'per_client_send_worker': 2}},
    {'pipeline': {'sender_queue_max_jobs': 3}},
    {'pipeline': {'completion_queue_max_jobs': 3}},
    {'pipeline': {'trainer_sender_decoupled': False}},
    {'random_sd': {'sample_ratio': .25}},
    {'prototype': {'alignment_weight_beta': 0.}},
    {'prototype': {'alignment_reduction': 'l2_squared'}},
])
def test_unsupported_controls_including_B_are_not_implemented(changes):
    with pytest.raises(ValueError):
        config(**changes)


def jobs(batch_count=24, *, version_start=0):
    rows = []
    for b in range(batch_count):
        g = b // 2
        size = min(2, batch_count - 2*g)
        v = version_start + g
        rows.append(dict(round=1, client_id=0, job_id=(1, 0, b), local_batch_index=b,
            pipeline_group_id=g, group_size=size, client_group_version=v,
            client_version_before=v, client_version_after=v+1, client_backward_version=v,
            client_optimizer_step_id=v, gradient_scale_to_group_mean=1/size,
            client_optimizer_step_performed=(b % 2 == size-1)))
    return rows


def test_group_validation_counts24jobs12steps_and_odd_last_group():
    from utils.na_sfl_train import validate_pipeline_groups
    assert validate_pipeline_groups(jobs(), require_zero_start=True) == 12
    assert validate_pipeline_groups(jobs(3), require_zero_start=True) == 2
    assert validate_pipeline_groups(jobs(version_start=12)) == 12
    with pytest.raises(ValueError):
        validate_pipeline_groups(jobs(version_start=12), require_zero_start=True)


@pytest.mark.parametrize('change', ['scale','version','early_step','duplicate','missing','nonconsecutive','size'])
def test_persisted_pipeline_invariants_reject_corruption(change):
    from utils.na_sfl_train import validate_pipeline_groups
    r = jobs()
    if change == 'scale': r[0]['gradient_scale_to_group_mean'] = 1.
    if change == 'version': r[1]['client_version_before'] += 1
    if change == 'early_step': r[0]['client_optimizer_step_performed'] = True
    if change == 'duplicate': r.append(copy.deepcopy(r[0]))
    if change == 'missing': r.pop(0)
    if change == 'nonconsecutive': r[0]['local_batch_index'] = 8
    if change == 'size': r[0]['group_size'] = 1
    with pytest.raises(ValueError):
        validate_pipeline_groups(r, require_zero_start=True)


def test_original_server_accepts_pipeline_without_changing_hybrid_semantics(monkeypatch):
    import torch
    from torch import nn
    from utils import sflv2
    from utils.na_sfl_prototype import PrototypeServer
    monkeypatch.setattr(sflv2, '_model_device', lambda _: torch.device('cpu'))
    c = config()
    c.update(boundary_shape_per_sample=(2,2,2))
    server = PrototypeServer(nn.Conv2d(2,10,1), c)
    assert server.hybrid_config == c['random_sd']
    assert server.batchnorm.policy == 'dual_running_stats_shared_affine'


def test_groups_cannot_start_at_missing_first_group_of_round():
    from utils.na_sfl_train import validate_pipeline_groups
    rows = jobs()[2:]
    with pytest.raises(ValueError):
        validate_pipeline_groups(rows)


def test_resume_checks_versions_per_client_not_only_global_sum(monkeypatch):
    from utils import na_sfl_train as trainer
    from utils.deit_async import async_trace_sha256
    rows = jobs()
    second = copy.deepcopy(rows)
    for row in second:
        row['client_id'] = 1
        row['job_id'] = (1, 1, row['local_batch_index'])
    rows += second
    for index, row in enumerate(rows):
        row.update(arrival_seq=index, server_version_before=index, server_version_after=index+1)
    boundary = dict(round=1, success=True, queue_empty=True, streams_completed=True,
        workers_alive=0, unresolved_futures=0, pending_graphs=0,
        trace_sha256=async_trace_sha256(rows),
        runtimecounts={k:48 for k in ('admitted_jobs','serviced_jobs','completed_jobs')})
    saved = dict(next_round=1, job_records=rows, config={'method':METHOD, 'n_clients':2},
        rounds=[{'round':1}], execution_boundaries=[boundary], aggregation_records=[{}],
        server_version=48, client_versions=[12,12], client_optimizers=[{'state':{}},{'state':{}}],
        auxiliary_optimizers=[])
    monkeypatch.setattr(trainer, 'validate_feature_memory_boundary', lambda _: None)
    trainer.validate_resume_metadata(saved)
    saved['client_versions'] = [13,11]
    with pytest.raises(ValueError):
        trainer.validate_resume_metadata(saved)


def test_c_manifest_explicitly_excludes_open_supervisor_console(tmp_path):
    import json
    from utils.sflv2_metrics import seal_manifest, sha256
    scientific = tmp_path/'summary.json'
    scientific.write_text('{"completed": true}\n')
    console = tmp_path/'supervisor.console.log'
    console.write_text('before seal\n')
    seal_manifest(tmp_path, {}, kind='pipeline_c', excluded_files=('supervisor.console.log',))
    console.write_text(console.read_text() + str(tmp_path) + '\n')
    saved = json.loads((tmp_path/'manifest.json').read_text())
    assert saved['artifacts'] == {'summary.json': sha256(scientific)}
    assert saved['excluded_artifacts'] == ['supervisor.console.log']


def test_legacy_manifest_keeps_existing_default_console_contract(tmp_path):
    import json
    from utils.sflv2_metrics import seal_manifest, sha256
    console = tmp_path/'supervisor.console.log'
    console.write_text('legacy\n')
    seal_manifest(tmp_path, {})
    saved = json.loads((tmp_path/'manifest.json').read_text())
    assert saved['artifacts']['supervisor.console.log'] == sha256(console)
    assert 'excluded_artifacts' not in saved

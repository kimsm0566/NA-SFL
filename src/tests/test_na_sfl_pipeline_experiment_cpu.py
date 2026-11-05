"""Synthetic supervisor contracts; no model execution and no training subprocess."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    from utils import na_sfl_pipeline_experiment
    return na_sfl_pipeline_experiment


def configs():
    from utils.na_sfl_config import DEFAULTS, load_config, PIPELINE_METHOD
    baseline = load_config('configs/na_sfl_proto_random_sd.yaml', {'rounds': 100})
    actual = load_config(overrides={**{k: baseline[k] for k in DEFAULTS},
        'method': PIPELINE_METHOD, 'result_path': '/tmp/pipeline-result'})
    return baseline, actual


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def seal(run):
    from utils.sflv2_metrics import sha256
    source = run / 'source_snapshot' / 'fixture.py'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('# synthetic artifact identity\n')
    hashes = {'fixture.py': sha256(source)}
    artifacts = {p.relative_to(run).as_posix(): sha256(p) for p in run.rglob('*')
                 if p.is_file() and p.name != 'manifest.json' and 'source_snapshot' not in p.parts}
    dump(run / 'manifest.json', {'artifacts': artifacts, 'source_sha256': hashes})
    return hashes


def pair(tmp_path):
    """One hand-timed synthetic round with240 Server jobs/120 Client steps."""
    from utils.na_sfl_network import rate_scenario
    from utils.na_sfl_hybrid import sample_indices
    from utils.deit_async import async_trace_sha256
    from utils.na_sfl_pipeline_artifacts import write_pipeline_artifacts
    baseline_config, actual = configs()
    baseline, run = tmp_path / 'reference', tmp_path / 'child'
    jobs = []
    for cid in range(10):
        for index in range(24):
            n, gid = (56 if index == 23 else 128), index // 2
            base = gid * 100.
            second = index % 2
            f = base + second * .5
            ul_start, ul_end = base + .2 + second * 2.1, base + 2.2 + second * 1.1
            slow = cid in (0, 5)
            selected = sample_indices(n, .1, 42, (1, cid, index)).tolist() if slow else []
            feature = (len(selected) + 1 if slow else n) * 128 * 16 * 16 * 4
            payload = feature + (16 + 12 * len(selected) if slow else 8 * n)
            gradient_payload = feature + (8 + 4 * len(selected) if slow else 0)
            row = dict(job_id=[1, cid, index], round=1, client_id=cid, local_batch_index=index,
                samples=n, labels=[cid] * n, sample_ids=list(range(cid * 3000 + index * 128,
                    cid * 3000 + index * 128 + n)), input_sha256=f'input-{cid}-{index}',
                job_kind='HYBRID_JOB' if slow else 'SD_JOB', proto_class_ids=[cid], proto_counts=[n],
                selected_sd_count=len(selected), selected_indices=selected,
                activation_bytes=feature, gradient_bytes=feature,
                sd_payload_bytes=payload, gradient_payload_bytes=gradient_payload,
                sd_wire_bytes=payload + 100, gradient_wire_bytes=gradient_payload + 100,
                upload_bytes=payload + 100, download_bytes=gradient_payload + 100,
                pipeline_group_id=gid, group_size=2, client_version_before=gid,
                client_group_version=gid, client_backward_version=gid,
                client_optimizer_step_id=gid, client_version_after=gid + 1,
                client_optimizer_step_performed=bool(second), gradient_scale_to_group_mean=.5,
                client_forward_start_s=f, forward_compute_end_s=f + .2, activation_ready_s=f + .3,
                forward_cuda_event_scope='model_forward', cuda_client_forward_start_s=f,
                cuda_client_forward_end_s=f + .2, cuda_event_origin_host_offset_s=0.,
                cuda_event_origin_host_uncertainty_s=.01,
                sd_send_start_s=ul_start, sd_receive_complete_s=ul_end,
                sendqueue_enqueue_s=ul_start - .01, sender_dequeue_s=ul_start,
                inflight_on_forward=second + 1, pending_graph_count=second + 1,
                max_pending_graphs=2, pending_graph_bytes=100 * (second + 1),
                pending_graph_count_after_backward=1 - second,
                max_sender_queue_depth=2, max_queued_cpu_bytes=200,
                max_completion_queue_depth=2, completion_queue_depth_after_backward=1 - second,
                client_idle_wait_s=.1 if second else 0., auxiliary_updated=False, gradient_mode='server')
            jobs.append(row)
    jobs.sort(key=lambda row: (row['sd_receive_complete_s'], row['client_id'], row['local_batch_index']))
    batches, clock = [], 0.
    for seq, row in enumerate(jobs):
        start = max(clock, row['sd_receive_complete_s']) + .001
        clock = start + .01
        row.update(arrival_seq=seq, service_seq=seq, enqueue_s=row['sd_receive_complete_s'],
            dequeue_s=start, server_start_s=start, server_end_s=clock,
            server_version_before=seq, server_version_after=seq + 1, server_batch_id=seq,
            server_model_compute_s=.003, gradient_available_s=clock + .02)
        batches.append(dict(round=1, batch_id=seq, server_version_before=seq,
            server_version_after=seq + 1, samples=row['samples'], sd_count=1, job_ids=[row['job_id']]))
    for row in jobs:
        same = [j for j in jobs if j['client_id'] == row['client_id'] and j['pipeline_group_id'] == row['pipeline_group_id']]
        complete = max(j['gradient_available_s'] for j in same) + .1
        row.update(group_complete_s=complete, client_complete_s=complete,
            group_optimizer_step_start_s=complete - .01, group_optimizer_step_end_s=complete)
    rates = {'round': 1, 'clients': [{'id': i, 'rate': 1} for i in range(10)], 'kernel_rates': [1] * 20}
    raw = dict(round=1, network_rates=rates, samples_seen=30000, server_updates=240,
        client_updates=120, training_wall_s_debug_only=1200., worker_drain_elapsed_s=1200.,
        test_accuracy=20., elapsed_wall_s=1200., gpu_peak_allocated_bytes=1000,
        gpu_peak_reserved_bytes=2000, global_prototype_stats=dict(round=1, version=1,
            sample_counts_by_class=[3000] * 10))
    transport = dict(frames_sent=480, frames_received=480, workers_alive=0, pending_sd=0,
        pending_gradients=0, reserved_sd=0, finished=True, aborted=False, error=None,
        wire_bytes_sent=sum(j['sd_wire_bytes'] + j['gradient_wire_bytes'] for j in jobs),
        pipeline=dict(max_outstanding_jobs_per_client=[2] * 10,
            max_sender_queue_depth_per_client=[2] * 10,
            max_gradient_receive_queue_depth_per_client=[2] * 10,
            max_queued_cpu_bytes_per_client=[200] * 10, queued_cpu_bytes=0,
            outstanding_jobs=0, sender_workers_per_client=1, receiver_workers_per_client=1))
    for kind in ('sd', 'gradient'):
        for direction in ('sent', 'received'):
            transport[kind + '_payload_bytes_' + direction] = sum(j[kind + '_payload_bytes'] for j in jobs)
    boundary = dict(round=1, success=True, queue_empty=True, streams_completed=True,
        workers_alive=0, pending_graphs=0, unresolved_futures=0,
        runtimecounts=dict(admitted_jobs=240, serviced_jobs=240, completed_jobs=240, serviced_batches=240),
        trace_sha256=async_trace_sha256(jobs), server_batch_trace_sha256=async_trace_sha256(batches),
        transport=transport, pipeline=dict(max_completion_queue_depth_per_client=[2] * 10,
            completion_queue_depth_per_client=[0] * 10, pending_completed_gradients=0, pending_job_rows=0))
    scenario = rate_scenario(actual['network'], 1)['sha256']
    base_summary = dict(completed=True, completed_rounds=100, run_dir=str(baseline),
        initial_model_hashes={'client': 'client', 'server': 'server', 'auxiliary': None},
        partition_sha256='partition', data_sha256='data',
        network_rate_scenario_sha256=rate_scenario(actual['network'], 100)['sha256'])
    for dest, cfg in ((baseline, baseline_config), (run, {**actual, 'rounds': 1, 'smoke_test': True})):
        dump(dest / 'config.json', cfg)
        dump(dest / 'round_metrics.json', [raw])
        dump(dest / 'summary.json', base_summary)
        (dest / 'per_job.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in jobs))
    sources = seal(baseline)
    dump(baseline / 'environment.json', {'gpu': {'uuid': 'GPU1'}, 'source_sha256': sources})
    seal(baseline)
    pipeline = write_pipeline_artifacts(run, [raw], jobs, config=actual)['pipeline']
    dump(run / 'summary.json', {**base_summary, 'completed_rounds': 1, 'run_dir': str(run),
        'samples_seen': 30000, 'server_updates': 240, 'auxiliary_heads': 0,
        'network_rate_scenario_sha256': scenario, 'pipeline': pipeline,
        'checkpoint': {'retained': False, 'finite': True},
        'server_batchnorm': {'policy': 'dual_running_stats_shared_affine',
            'forward_calls': {'sd': 240, 'proto': 240}, 'running_updates': {'sd': 240, 'proto': 240},
            'running_counters': {str(i): {'sd': 240, 'proto': 240} for i in range(10)}}})
    dump(run / 'environment.json', {'gpu': {'uuid': 'GPU1'}, 'source_sha256': sources})
    dump(run / 'execution_boundaries.json', [boundary])
    dump(run / 'server_batches.json', batches)
    dump(run / 'aggregation.json', [{'round': 1, 'weights': [3000] * 10}])
    dump(run / 'network/network_metadata.json', {'closed': True, 'cleanup_errors': [], 'model_traffic': []})
    seal(run)
    return run, baseline, sources


def test_only_method_pipeline_and_output_device_differ_from_existing_v5():
    baseline, actual = configs()
    module().validate_config(baseline, actual)
    for changes in ({'lr': .002}, {'rounds': 99}, {'seed': 43},
                    {'pipeline': {**actual['pipeline'], 'max_inflight_per_client': 1}}):
        with pytest.raises(ValueError):
            module().validate_config(baseline, {**actual, **changes})


def test_dispatch_has_only_smoke_then_fresh_c_main_and_failure_blocks_main():
    calls = []
    def launch(name, rounds, smoke):
        calls.append((name, rounds, smoke))
        return Path(name)
    assert module().launch_sequence(launch) == (Path('smoke_test'), Path('run_100round'))
    assert calls == [('smoke_test', 2, True), ('run_100round', 100, False)]
    calls.clear()
    def failed(name, rounds, smoke):
        calls.append(name)
        raise ValueError('Smoke audit failed')
    with pytest.raises(ValueError, match='Smoke audit'):
        module().launch_sequence(failed)
    assert calls == ['smoke_test']


def test_complete_synthetic_pipeline_audit_preserves_both_inputs(tmp_path):
    run, baseline, sources = pair(tmp_path)
    before = {p: p.read_bytes() for folder in (run, baseline) for p in folder.rglob('*') if p.is_file()}
    result = module().audit_run(run, 1, baseline, sources, 'GPU1', smoke=True)
    assert result['passed'] and result['client_optimizer_steps'] == 120
    assert result['server_updates'] == 240 and result['client_visits'] == 30000
    assert result['actual_sd_count'] == 24610 and result['selected_sd_count'] == 610
    assert result['pipeline']['client_forward_uplink_overlap_lower_bound_s'] > 0
    assert result['pipeline']['max_completion_queue_depth'] == 2
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize('mutation', ['host_scope', 'no_overlap', 'completion_leak',
    'completion_peak', 'tcp_bytes', 'missing_job', 'input_identity', 'extra_client_step'])
def test_audit_rejects_broken_pipeline_contracts(tmp_path, mutation):
    run, baseline, sources = pair(tmp_path)
    if mutation in ('completion_leak', 'completion_peak', 'tcp_bytes'):
        rows = json.loads((run / 'execution_boundaries.json').read_text())
        if mutation == 'completion_leak':
            rows[0]['pipeline']['pending_completed_gradients'] = 1
        elif mutation == 'completion_peak':
            rows[0]['pipeline']['max_completion_queue_depth_per_client'][0] = 3
        else:
            rows[0]['transport']['sd_payload_bytes_received'] += 1
        dump(run / 'execution_boundaries.json', rows)
    else:
        jobs = [json.loads(line) for line in (run / 'per_job.jsonl').read_text().splitlines()]
        if mutation == 'host_scope':
            jobs[0]['forward_cuda_event_scope'] = 'cpu_test_adapter'
        elif mutation == 'no_overlap':
            for row in jobs:
                row['cuda_client_forward_start_s'] = 0.
                row['cuda_client_forward_end_s'] = 0.
        elif mutation == 'missing_job':
            jobs.pop()
        elif mutation == 'input_identity':
            jobs[0]['input_sha256'] = 'wrong-input'
        else:
            jobs[0]['client_optimizer_step_performed'] = True
        (run / 'per_job.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in jobs))
    seal(run)
    with pytest.raises(ValueError):
        module().audit_run(run, 1, baseline, sources, 'GPU1', smoke=True)


def test_reference_check_verifies_frozen_sources_in_addition_to_artifacts(tmp_path):
    run, baseline, sources = pair(tmp_path)
    manifest = json.loads((baseline / 'manifest.json').read_text())
    module().check_reference(baseline, manifest)
    (baseline / 'source_snapshot/fixture.py').write_text('# altered\n')
    with pytest.raises(ValueError):
        module().check_reference(baseline, manifest)


def test_legacy_derived_job_aliases_do_not_invalidate_original_engine_trace(tmp_path):
    run, baseline, sources = pair(tmp_path)
    jobs = [json.loads(line) for line in (run / 'per_job.jsonl').read_text().splitlines()]
    for row in jobs:
        row['server_arrival_ts'] = row['sd_receive_complete_s']
        row['upload_done_ts'] = row['sd_receive_complete_s']
    (run / 'per_job.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in jobs))
    seal(run)
    assert module().audit_run(run, 1, baseline, sources, 'GPU1', smoke=True)['passed']


def test_existing_a_metrics_are_labelled_reference_and_keep_unknown_pipeline_fields(tmp_path):
    from utils.na_sfl_pipeline_artifacts import write_pipeline_artifacts
    baseline, actual = configs()
    result = write_pipeline_artifacts(tmp_path, [{'round': 1, 'server_updates': 1}],
        [{'job_id': [1, 0, 0], 'samples': 128}], config=baseline)['pipeline']
    assert result['mode'] == 'A_w1_step1'
    assert result['client_forward_uplink_overlap_s'] is None
    assert result['client_optimizer_steps'] is None


def test_cpu_child_failure_records_failure_and_never_creates_main_config(tmp_path, monkeypatch):
    run, baseline, sources = pair(tmp_path)
    rows = json.loads((baseline / 'round_metrics.json').read_text())
    dump(baseline / 'round_metrics.json', [{**rows[0], 'round': rid} for rid in range(1, 101)])
    seal(baseline)
    source = tmp_path / 'synthetic_source'
    source.mkdir()
    (source / 'run_exp_main.py').write_text('raise SystemExit(7)\n')
    dump(source / 'results/checkpoint_registry.json', {'final_checkpoints': [], 'global_limit': 5})
    experiment = module()
    monkeypatch.setattr(experiment, 'SOURCE_ROOT', source)
    monkeypatch.setattr(experiment, 'gpu_snapshot', lambda _: {'uuid': 'GPU1',
        'name': 'Quadro RTX 5000', 'compute_processes': [], 'free_bytes': 100, 'total_bytes': 100})
    actual = configs()[1]
    actual['result_path'] = str(tmp_path / 'launch')
    with pytest.raises(RuntimeError, match='smoke_test failed with exit 7'):
        experiment.run_pipeline_experiment(SimpleNamespace(config=actual, baseline_reference=baseline))
    state = json.loads((tmp_path / 'launch/status.json').read_text())
    assert state['status'] == 'failed' and state['active_child_pid'] is None
    assert not state['completed'] and state['active_condition'] == 'smoke_test'
    assert not (tmp_path / 'launch/inputs/run_100round.yaml').exists()


def test_comparison_writes_derived_reports_and_preserves_reference(tmp_path):
    from utils.na_sfl_config import HYBRID_METHOD, PIPELINE_METHOD
    baseline_cfg, main_cfg = configs()
    base, main, output = tmp_path / 'a', tmp_path / 'c', tmp_path / 'reports'
    for folder, cfg, accuracy, step in ((base, baseline_cfg, 60., 2), (main, main_cfg, 65., 1)):
        rounds = [dict(round=rid, test_accuracy=accuracy, elapsed_wall_s=10. * rid,
            training_wall_s_debug_only=9., worker_drain_elapsed_s=9., server_updates=2,
            client_updates=step, gpu_peak_allocated_bytes=1000, gpu_peak_reserved_bytes=2000)
            for rid in range(1, 101)]
        jobs = []
        for rid in range(1, 101):
            for index in (0, 1):
                row = dict(job_id=[rid, 0, index], round=rid, client_id=0, local_batch_index=index,
                    samples=128, job_kind='SD_JOB', proto_class_ids=[0], upload_bytes=20,
                    download_bytes=30, activation_bytes=8, gradient_bytes=8,
                    sd_payload_bytes=16, gradient_payload_bytes=8,
                    sd_receive_complete_s=1. + index, server_start_s=2. + index,
                    server_end_s=2.5 + index, server_model_compute_s=.1)
                if cfg['method'] == PIPELINE_METHOD:
                    row.update(pipeline_group_id=0, group_size=2,
                        client_optimizer_step_performed=bool(index), client_idle_wait_s=0.)
                jobs.append(row)
        dump(folder / 'config.json', cfg)
        dump(folder / 'round_metrics.json', rounds)
        dump(folder / 'summary.json', {'launch_wall_s_debug_only': 1005.,
            'initial_evaluation': {'test_accuracy': 10., 'elapsed_wall_s': 0.},
            'server_updates': 200})
        dump(folder / 'environment.json', {'gpu': {'uuid': 'GPU1'}})
        dump(folder / 'network/network_metadata.json', {'model_traffic': [
            {'direction': 'upload', 'wire_bytes': 10, 'payload_bytes': 8},
            {'direction': 'download', 'wire_bytes': 20, 'payload_bytes': 16}]})
        (folder / 'per_job.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in jobs))
    before = {p: p.read_bytes() for folder in (base, main) for p in folder.rglob('*') if p.is_file()}
    records = module().render_results(output, base, main)
    assert records[0]['client_optimizer_steps'] == 200
    assert records[1]['client_optimizer_steps'] == 100
    assert records[1]['delta_last10_vs_v5_pp'] == 5.
    assert records[0]['client_forward_uplink_overlap_s'] is None
    assert records[1]['total_upload_bytes_including_models'] == 4010
    assert records[0]['round_train_mean_s'] == records[0]['round_train_median_s'] == 9.
    assert records[1]['round_wall_mean_s'] is None
    assert records[0]['synchronization_wall_total_s'] is None
    assert records[0]['server_busy_wall_percent'] == pytest.approx(100. / 9.)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert (output / 'comparison.csv').is_file() and (output / 'comparison.json').is_file()
    for stem in ('accuracy_by_round', 'accuracy_by_elapsed', 'queue_busy_idle', 'overlap_memory_steps'):
        assert (output / (stem + '.png')).is_file() and (output / (stem + '.pdf')).is_file()
    assert 'combined' in (output / 'README.md').read_text()


def test_timing_record_uses_measured_mean_median_and_sync_totals(tmp_path):
    rows = [{'round': i, 'training_wall_s_debug_only': train, 'round_wall_s_debug_only': wall}
            for i, train, wall in [(1, 1., 2.), (2, 3., 4.), (3, 8., 9.)]]
    dump(tmp_path / 'aggregation.json', [{'round': i, 'aggregation_wall_s': value}
        for i, value in [(1, .5), (2, 1.), (3, 1.5)]])
    result = module().timing_record(tmp_path, rows)
    assert result == {'round_train_mean_s': 4., 'round_train_median_s': 3.,
        'round_wall_mean_s': 5., 'round_wall_median_s': 4.,
        'synchronization_wall_mean_s': 1., 'synchronization_wall_total_s': 3.}


def test_incomplete_round_or_sync_timings_remain_unknown(tmp_path):
    rows = [{'round': 1, 'training_wall_s_debug_only': 1.}, {'round': 2}]
    dump(tmp_path / 'aggregation.json', [{'round': 1, 'aggregation_wall_s': .5}, {'round': 2}])
    assert all(value is None for value in module().timing_record(tmp_path, rows).values())

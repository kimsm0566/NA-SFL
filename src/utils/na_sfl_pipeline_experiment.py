"""C-only Smoke/main orchestration through the existing single training CLI."""
from copy import deepcopy
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import subprocess
from statistics import median
import sys
from tempfile import TemporaryDirectory

import yaml

from utils.deit_artifacts import csv_file, gpu_snapshot, json_file
from utils.deit_async import async_trace_sha256
from utils.na_sfl_config import DEFAULTS, HYBRID_METHOD, PIPELINE_METHOD, PIPELINE_DEFAULTS, load_config
from utils.na_sfl_bn_experiments import (_now, last10_mean, controlled_termination,
    owned_stage_process, wait_stage_process, _retention_audit)
from utils.na_sfl_hybrid_experiment import _validate_run, _communication, accuracy_record, model_traffic
from utils.na_sfl_pipeline_artifacts import write_pipeline_artifacts
from utils.na_sfl_train import validate_pipeline_groups
from utils.na_sfl_batch import validate_server_batches
from utils.na_sfl_network import rate_scenario
from utils.sflv2_metrics import sha256, snapshot_sources, check_sources_unchanged, seal_manifest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return json.loads(Path(path).read_text())


def _jobs(run):
    with (Path(run) / 'per_job.jsonl').open() as stream:
        return [json.loads(line) for line in stream]


def validate_config(baseline, actual):
    p = baseline['prototype']
    if (baseline['method'] != HYBRID_METHOD or baseline['rounds'] != 100 or baseline['seed'] != 42
            or baseline['bn_policy'] != 'dual_running_stats_shared_affine'
            or baseline['random_sd'] != {'sample_ratio': .1, 'sample_seed': 42}
            or (p['fast_sd_weight'], p['fast_proto_weight'], p['slow_proto_weight'],
                p['alignment_weight_beta'], p['alignment_reduction']) != (.8, .2, 1., .1, 'mse')
            or baseline['network'] is None or baseline['smoke_test']
            or baseline['max_batches_per_client'] is not None):
        raise ValueError('Pipeline reference must be completed original v5 / Dual BN / SD10% / seed42')
    expected = deepcopy(baseline)
    expected.update(method=PIPELINE_METHOD, pipeline=dict(PIPELINE_DEFAULTS),
        result_path=actual['result_path'], device=actual['device'])
    if expected != actual or actual['pipeline'] != PIPELINE_DEFAULTS:
        changed = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
        raise ValueError('C config differs beyond pipeline/method/output/device: ' + str(changed))


def launch_sequence(launch):
    smoke = launch('smoke_test', 2, True)
    main = launch('run_100round', 100, False)
    return smoke, main


def check_reference(run, manifest):
    """Verify both original artifacts and frozen sources; perform no writes."""
    run = Path(run)
    for name, digest in manifest['artifacts'].items():
        if sha256(run / name) != digest:
            raise ValueError('v5 reference artifact changed: ' + name)
    for name, digest in manifest['source_sha256'].items():
        if sha256(run / 'source_snapshot' / name) != digest:
            raise ValueError('v5 reference frozen source changed: ' + name)


def _bounded(values, name, *, count=10):
    if (not isinstance(values, list) or len(values) != count
            or any(type(value) is not int or not 0 <= value <= 2 for value in values)):
        raise ValueError('C bounded queue/window audit failed: ' + name)


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def audit_run(run, rounds, baseline_run, sources, gpu_uuid, *, smoke=False):
    """Independent event audit; recomputation writes only disposable CPU reports."""
    run, baseline_run = Path(run), Path(baseline_run)
    baseline = {**_read(baseline_run / 'summary.json'), 'run_dir': str(baseline_run)}
    summary = _validate_run(run, rounds, baseline, gpu_uuid, sources, smoke=smoke)
    rows, jobs = _read(run / 'round_metrics.json'), _jobs(run)
    config = _read(run / 'config.json')
    baseline_config = load_config(overrides={k: v for k, v in _read(baseline_run / 'config.json').items() if k in DEFAULTS})
    stage = load_config(overrides={**{k: v for k, v in config.items() if k in DEFAULTS},
        'rounds': 100, 'smoke_test': False})
    validate_config(baseline_config, stage)
    if summary.get('auxiliary_heads') != 0 or validate_pipeline_groups(jobs, require_zero_start=True) != 120 * rounds:
        raise ValueError('C must have zero Aux heads and120 Client optimizer steps/round')
    references = {}
    for row in _jobs(baseline_run):
        if row['round'] <= rounds:
            references[tuple(row['job_id'])] = (row['input_sha256'], row['labels'], row['sample_ids'])
    for row in jobs:
        if references.pop(tuple(row['job_id']), None) != (row['input_sha256'], row['labels'], row['sample_ids']):
            raise ValueError('C input tensor hash, labels or sample IDs differ from original v5')
        if (row.get('forward_cuda_event_scope') != 'model_forward'
                or row.get('auxiliary_updated') is not False or row.get('gradient_mode') != 'server'):
            raise ValueError('C requires actual model-only CUDA Forward events and Server gradients')
        for field in ('inflight_on_forward', 'pending_graph_count', 'max_pending_graphs',
                      'pending_graph_count_after_backward', 'max_sender_queue_depth',
                      'max_completion_queue_depth', 'completion_queue_depth_after_backward'):
            value = row.get(field)
            if type(value) is not int or not 0 <= value <= 2:
                raise ValueError('C missing or unbounded per-job graph/window/queue: ' + field)
    if references:
        raise ValueError('C omitted reference jobs')
    batches = _read(run / 'server_batches.json')
    if validate_server_batches(jobs, batches, config['server_max_batch_samples'], 'individual') != 240 * rounds:
        raise ValueError('C must retain240 independent Server updates/round')
    boundaries = _read(run / 'execution_boundaries.json')
    aggregation = _read(run / 'aggregation.json')
    if len(boundaries) != rounds or len(aggregation) != rounds:
        raise ValueError('C missing drain or FedAvg boundaries')
    overlap_pairs = 0
    actual_sd = selected_sd = 0
    for rid, boundary, aggregate in zip(range(1, rounds + 1), boundaries, aggregation):
        own = [row for row in jobs if row['round'] == rid]
        own_batches = [row for row in batches if row['round'] == rid]
        raw = rows[rid - 1]
        if (len(own) != 240 or raw.get('client_updates') != 120 or raw['server_updates'] != 240
                or aggregate['round'] != rid or aggregate['weights'] != [3000] * 10
                or [row['arrival_seq'] for row in own] != list(range(240))
                or [row['server_version_before'] for row in own] != list(range(240 * (rid - 1), 240 * rid))
                or [row['sd_receive_complete_s'] for row in own] != sorted(row['sd_receive_complete_s'] for row in own)):
            raise ValueError('C visits, FCFS order, Client steps or visit-weighted FedAvg differs')
        for cid in range(10):
            client = sorted((row for row in own if row['client_id'] == cid), key=lambda row: row['local_batch_index'])
            if (len(client) != 24 or sum(row['samples'] for row in client) != 3000
                    or sorted({row['pipeline_group_id'] for row in client}) != list(range(12))
                    or sum(row['client_optimizer_step_performed'] for row in client) != 12):
                raise ValueError('C Client must retain24 jobs/3000 visits/12 groups')
            for first, second in zip(client[::2], client[1::2]):
                start = second.get('cuda_event_origin_host_offset_s')
                error = second.get('cuda_event_origin_host_uncertainty_s')
                if start is None or error is None or error < 0:
                    raise ValueError('missing C CUDA/host clock calibration')
                low = start + second['cuda_client_forward_start_s'] + error
                high = start + second['cuda_client_forward_end_s'] - error
                if (second['client_forward_start_s'] < first['sd_receive_complete_s']
                        and max(low, first['sd_send_start_s']) < min(high, first['sd_receive_complete_s'])):
                    overlap_pairs += 1
        proto = raw.get('global_prototype_stats') or {}
        if (proto.get('round') != rid or proto.get('version') != rid
                or sum(proto.get('sample_counts_by_class', [])) != 30000):
            raise ValueError('C Global Prototype must update once per completed30000-visit round')
        # Legacy per_job.jsonl adds reporting aliases to raw engine records.
        # Its JSON therefore cannot recreate the original raw-job trace hash.
        # The original digest remains covered by the verified stage manifest;
        # Server batch JSON is raw and can still be independently re-hashed.
        trace = boundary.get('trace_sha256')
        if (boundary.get('round') != rid or not boundary.get('success') or not boundary.get('queue_empty')
                or not boundary.get('streams_completed')
                or any(boundary.get(key) != 0 for key in ('workers_alive', 'pending_graphs', 'unresolved_futures'))
                or not isinstance(trace, str) or len(trace) != 64 or any(c not in '0123456789abcdef' for c in trace)
                or boundary.get('server_batch_trace_sha256') != async_trace_sha256(own_batches)
                or any(boundary['runtimecounts'].get(key) != 240 for key in
                       ('admitted_jobs', 'serviced_jobs', 'completed_jobs', 'serviced_batches'))):
            raise ValueError('C actor/stream/trace drain failed')
        completion = boundary.get('pipeline') or {}
        _bounded(completion.get('max_completion_queue_depth_per_client'), 'actual completion peaks')
        if (completion.get('completion_queue_depth_per_client') != [0] * 10
                or completion.get('pending_completed_gradients') != 0 or completion.get('pending_job_rows') != 0):
            raise ValueError('C completed-gradient queue did not drain')
        transport = boundary['transport']
        if (transport.get('frames_sent') != 480 or transport.get('frames_received') != 480
                or any(transport.get(key) != 0 for key in ('workers_alive', 'pending_sd', 'pending_gradients', 'reserved_sd'))
                or transport.get('finished') is not True or transport.get('aborted') or transport.get('error')):
            raise ValueError('C TCP worker/frame drain failed')
        peaks = transport.get('pipeline') or {}
        for field in ('max_outstanding_jobs_per_client', 'max_sender_queue_depth_per_client',
                      'max_gradient_receive_queue_depth_per_client'):
            _bounded(peaks.get(field), field)
        if (peaks.get('queued_cpu_bytes') != 0 or peaks.get('outstanding_jobs') != 0
                or peaks.get('sender_workers_per_client') != 1 or peaks.get('receiver_workers_per_client') != 1):
            raise ValueError('C Sender/Receiver must remain single and fully drained')
        for kind in ('sd', 'gradient'):
            total = sum(row[kind + '_payload_bytes'] for row in own)
            if any(transport.get(kind + '_payload_bytes_' + direction) != total for direction in ('sent', 'received')):
                raise ValueError('C actual TCP payload byte totals differ')
        if transport.get('wire_bytes_sent') != sum(row['sd_wire_bytes'] + row['gradient_wire_bytes'] for row in own):
            raise ValueError('C actual TCP application wire byte totals differ')
        actual_sd += sum(row['samples'] if row['job_kind'] == 'SD_JOB' else row['selected_sd_count'] for row in own)
        selected_sd += sum(row.get('selected_sd_count', 0) for row in own if row['job_kind'] == 'HYBRID_JOB')
    if actual_sd != 24610 * rounds or selected_sd != 610 * rounds:
        raise ValueError('C original full Fast SD / Slow SD10% counts changed')
    with TemporaryDirectory(prefix='hsfl-pipeline-audit-') as temporary:
        metrics = write_pipeline_artifacts(temporary, rows, jobs, config=config)['pipeline']
    for field in ('jobs', 'client_optimizer_steps', 'server_updates', 'max_inflight', 'max_pending_graphs',
                  'client_forward_uplink_overlap_s', 'client_forward_uplink_overlap_lower_bound_s',
                  'client_forward_uplink_overlap_clock'):
        if summary.get('pipeline', {}).get(field) != metrics[field]:
            raise ValueError('persisted C metrics differ from raw event recomputation: ' + field)
    if (metrics['max_inflight'] != 2 or metrics['max_pending_graphs'] != 2
            or metrics['client_forward_uplink_overlap_clock'] != 'aligned_cuda_events'
            or not _positive(metrics['client_forward_uplink_overlap_lower_bound_s']) or not overlap_pairs):
        raise ValueError('C did not demonstrate bounded real CUDA Forward(B2)/UL(B1) overlap')
    if smoke and any((run / 'checkpoints').glob('*.pt')):
        raise ValueError('C Smoke checkpoint was not discarded')
    return {'passed': True, 'run_dir': str(run), 'rounds': rounds, 'jobs': len(jobs),
        'client_visits': 30000 * rounds, 'client_optimizer_steps': 120 * rounds,
        'server_updates': 240 * rounds, 'actual_sd_count': actual_sd, 'selected_sd_count': selected_sd,
        'overlapping_second_forward_first_uplink_pairs': overlap_pairs, 'pipeline': metrics,
        'job_trace_scope': 'original raw digest preserved by manifest; legacy reporting aliases prevent direct re-hash',
        'scope': 'CPU audit of completed actual execution; two rounds do not establish long-run memory stability'}


def _registry_snapshot(source):
    path = source / 'results/checkpoint_registry.json'
    raw = path.read_bytes()
    registry = json.loads(raw)
    entries = registry['final_checkpoints']
    if len(entries) > 5:
        raise ValueError('checkpoint registry exceeds project latest-five limit')
    for entry in entries:
        checkpoint = Path(entry['path'])
        if (not checkpoint.is_file() or checkpoint.stat().st_size != entry['bytes']
                or sha256(checkpoint) != entry['sha256']):
            raise ValueError('retained checkpoint identity differs before/after Smoke')
    return raw


def timing_record(run, rows):
    """Measured round statistics and actual FedAvg synchronization wall only."""
    def measured(records, field):
        values = [row.get(field) for row in records]
        if not values or any(value is None for value in values):
            return None
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('invalid measured timing: ' + field)
        return values
    result = {}
    for prefix, field in [('round_train', 'training_wall_s_debug_only'),
                          ('round_wall', 'round_wall_s_debug_only')]:
        values = measured(rows, field)
        result[prefix + '_mean_s'] = sum(values) / len(values) if values is not None else None
        result[prefix + '_median_s'] = median(values) if values is not None else None
    path = Path(run) / 'aggregation.json'
    aggregation = _read(path) if path.is_file() else []
    values = (measured(aggregation, 'aggregation_wall_s')
        if [row.get('round') for row in aggregation] == [row['round'] for row in rows] else None)
    result.update(synchronization_wall_mean_s=sum(values) / len(values) if values is not None else None,
        synchronization_wall_total_s=sum(values) if values is not None else None)
    return result


def render_results(root, baseline_run, main_run):
    """Derive A/C reports into new folders; never write inside either source run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root, baseline_run, main_run = Path(root), Path(baseline_run), Path(main_run)
    root.mkdir(parents=True, exist_ok=True)
    records, series, traffic_rows = [], [], []
    metric_fields = ('training_wall_s', 'server_busy_wall_s', 'server_idle_wall_s', 'server_cuda_model_busy_s',
        'server_busy_wall_percent', 'server_idle_wall_percent',
        'server_queue_wait_mean_s', 'server_queue_wait_p95_s', 'server_queue_wait_max_s',
        'server_queue_depth_mean', 'server_queue_depth_max', 'client_idle_wait_s', 'gradient_pending_s',
        'round_end_wait_s', 'sender_queue_wait_s', 'client_forward_uplink_overlap_s',
        'client_forward_uplink_overlap_lower_bound_s', 'client_forward_uplink_overlap_host_phase_s',
        'client_forward_uplink_overlap_paced_s', 'client_forward_uplink_overlap_clock',
        'max_inflight', 'max_pending_graphs', 'max_pending_graph_boundary_bytes', 'max_sendqueue_depth',
        'max_sendqueue_bytes', 'max_completion_queue_depth', 'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes')
    for name, run, destination in [('A_existing_v5', baseline_run, 'baseline_artifacts'),
                                    ('C_w2_accum2', main_run, 'pipeline_artifacts')]:
        summary, rows, config = _read(run / 'summary.json'), _read(run / 'round_metrics.json'), _read(run / 'config.json')
        jobs = _jobs(run)
        metrics = write_pipeline_artifacts(root / destination, rows, jobs, config=config,
            initial_evaluation=summary.get('initial_evaluation'), render=True)['pipeline']
        record = accuracy_record(name, run, summary, rows)
        record.update(timing_record(run, rows))
        record.update({field: metrics[field] for field in metric_fields})
        record['client_optimizer_steps'] = (metrics['client_optimizer_steps'] if name.startswith('C_')
            else sum(row['client_updates'] for row in rows) if all(row.get('client_updates') is not None for row in rows) else None)
        record['server_updates'] = metrics['server_updates']
        totals = _communication(run)
        record.update(totals['all'])
        record.update(model_traffic(run))
        record.update(gpu_uuid=_read(run / 'environment.json')['gpu']['uuid'], mode=metrics['mode'])
        for direction in ('upload', 'download'):
            record['total_' + direction + '_bytes_including_models'] = record[direction + '_bytes'] + record['model_' + direction + '_bytes']
        record['delta_last10_vs_v5_pp'] = float(Decimal(record['last10_mean_exact']) - Decimal(
            records[0]['last10_mean_exact'] if records else record['last10_mean_exact']))
        records.append(record)
        series.append((name, rows, metrics['round_metrics']))
        traffic_rows.extend({'condition': name, 'client_group': group, **values} for group, values in totals.items())
    json_file(root / 'comparison.json', records)
    csv_file(root / 'comparison.csv', records)
    csv_file(root / 'communication_bytes.csv', traffic_rows)
    def save(fig, stem):
        for suffix in ('.png', '.pdf'):
            fig.savefig(root / (stem + suffix), dpi=160)
        plt.close(fig)
    for stem, key, xlabel, scale in [('accuracy_by_round', 'round', 'Round', 1),
        ('accuracy_by_elapsed', 'elapsed_wall_s', 'Training through evaluation (hours)', 3600)]:
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        for name, rows, metrics in series:
            ax.plot([row[key] / scale for row in rows], [row['test_accuracy'] for row in rows], label=name)
        ax.axhline(70, color='gray', linestyle=':')
        ax.set(xlabel=xlabel, ylabel='Original-SD test accuracy (%)')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        save(fig, stem)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, fields, title in zip(axes.flat,
        [('server_queue_wait_mean_s', 'server_queue_wait_p95_s'), ('server_busy_wall_s', 'server_idle_wall_s'),
         ('client_idle_wait_s',), ('server_queue_depth_mean',)],
        ['Receive-to-service queue (seconds)', 'Server service wall (seconds)',
         'Trainer Future wait (client-seconds)', 'Time-weighted waiting jobs']):
        for name, rows, metrics in series:
            for field in fields:
                points = [row for row in metrics if row.get(field) is not None]
                if points:
                    ax.plot([row['round'] for row in points], [row[field] for row in points], label=name + ' ' + field)
        ax.set(xlabel='Round', title=title); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=6)
    save(fig, 'queue_busy_idle')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fields = [('client_forward_uplink_overlap_lower_bound_s', 'Conservative aligned CUDA overlap (client-seconds)'),
        ('gpu_peak_allocated_bytes', 'Actual allocated GPU peak (bytes)'),
        ('max_pending_graph_boundary_bytes', 'Retained boundary-byte lower bound'),
        ('client_optimizer_steps', 'Client optimizer steps per round')]
    for ax, (field, title) in zip(axes.flat, fields):
        for name, rows, metrics in series:
            data = rows if field == 'client_optimizer_steps' and name.startswith('A_') else metrics
            key = 'client_updates' if data is rows and field == 'client_optimizer_steps' else field
            points = [row for row in data if row.get(key) is not None]
            if points:
                ax.plot([row['round'] for row in points], [row[key] for row in points], label=name)
        ax.set(xlabel='Round', title=title); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=8)
    save(fig, 'overlap_memory_steps')
    lines = ['# Existing v5 A versus C=w2_accum2', '',
        'Only C is newly trained. A is the completed original v5 reference. No additional comparison condition is run.',
        'Same GPU UUID / seed42 / initial models / data / partition / variable links / selected SD indices.',
        'This is the combined effect of pipelining and accumulated Client update frequency; their causal effects are not separated.', '',
        '| Condition | Final | Best | Last10 | Delta(pp) | TTA70(s) | Training/evaluation(s) | Client steps | Server steps | UL incl models(bytes) | DL incl models(bytes) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"|{row['name']}|{row['final_accuracy']:.2f}|{row['best_accuracy']:.2f}|{row['last10_mean']:.3f}|"
            f"{row['delta_last10_vs_v5_pp']:.3f}|{row['tta_70']}|{row['training_through_evaluation_s']:.3f}|"
            f"{row['client_optimizer_steps']}|{row['server_updates']}|{row['total_upload_bytes_including_models']}|"
            f"{row['total_download_bytes_including_models']}|")
    lines += ['', 'Missing measurements stay null; original A lacks calibrated CUDA/pipeline-specific events.',
        'Round train/wall mean and median use their recorded durations. Synchronization uses measured aggregation_wall_s '
        '(model upload, FedAvg, broadcast and local optimizer reset), with no inferred missing duration.',
        'Queue uses full SD receipt to service start. Server wall idle is not measured GPU utilization.',
        'Graph bytes are a retained boundary tensor lower bound, separate from actual GPU allocation/reservation.',
        'Wire bytes include application framing and actual model/FedAvg traffic; exclude TCP/IP/link overhead.',
        'TTA70 is the first evaluated crossing, not sustained accuracy. Single seed and live FCFS ordering apply.', '',
        '![Accuracy by round](accuracy_by_round.png)', '', '![Accuracy by time](accuracy_by_elapsed.png)', '',
        '![Queue and service idle](queue_busy_idle.png)', '', '![Overlap memory steps](overlap_memory_steps.png)']
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
    return records


def run_pipeline_experiment(args):
    from utils.na_sfl_run import validate_training_gpu
    config, source = args.config, SOURCE_ROOT
    baseline_run = Path(args.baseline_reference).resolve()
    baseline = _read(baseline_run / 'summary.json')
    baseline_config = load_config(overrides={k: v for k, v in _read(baseline_run / 'config.json').items() if k in DEFAULTS})
    validate_config(baseline_config, config)
    if not baseline['completed'] or baseline['completed_rounds'] != 100:
        raise ValueError('C requires an existing completed100round v5 reference')
    last10_mean(_read(baseline_run / 'round_metrics.json'))
    manifest = _read(baseline_run / 'manifest.json')
    check_reference(baseline_run, manifest)
    gpu = gpu_snapshot(config['device'])
    validate_training_gpu(gpu, allow_unprofiled=config['allow_unprofiled_gpu'])
    if gpu['uuid'] != _read(baseline_run / 'environment.json')['gpu']['uuid']:
        raise ValueError('C must use original v5 GPU1 UUID')
    scenario = rate_scenario(config['network'], 100)
    if scenario['sha256'] != baseline['network_rate_scenario_sha256']:
        raise ValueError('C variable network scenario differs from v5')
    root = Path(config['result_path']).resolve()
    if (root == baseline_run or root.is_relative_to(baseline_run) or baseline_run.is_relative_to(root)
            or (root / 'status.json').exists() or (root / 'manifest.json').exists()
            or any((root / name).exists() for name in ('smoke_test', 'run_100round'))):
        raise ValueError('C destination overlaps an existing launch or reference')
    root.mkdir(parents=True, exist_ok=True)
    sources = snapshot_sources(root)
    json_file(root / 'source_sha256.json', sources)
    json_file(root / 'baseline_reference.json', {'run_dir': str(baseline_run),
        'artifacts': manifest['artifacts'], 'source_sha256': manifest['source_sha256']})
    json_file(root / 'network_rate_scenario.json', scenario)
    (root / 'inputs').mkdir(exist_ok=True)
    state = {'status': 'running', 'completed': False, 'started_at': _now(), 'supervisor_pid': os.getpid(),
        'condition': 'C_w2_accum2', 'reference_condition': 'A_existing_v5',
        'gpu_uuid': gpu['uuid'], 'active_condition': None, 'active_child_pid': None, 'active_rounds_completed': 0}
    def status():
        state['updated_at'] = _now()
        json_file(root / 'status.json', state)
    def launch(name, rounds, smoke):
        check_sources_unchanged(sources)
        check_reference(baseline_run, manifest)
        registry_before = _registry_snapshot(source) if smoke else None
        stage_config = {**{key: config[key] for key in DEFAULTS}, 'rounds': rounds,
            'smoke_test': smoke, 'result_path': str(root / name)}
        path = root / 'inputs' / (name + '.yaml')
        path.write_text(yaml.safe_dump(stage_config, sort_keys=True))
        command = [sys.executable, '-u', str(source / 'run_exp_main.py'), '--config', str(path)]
        json_file(root / (name + '_launch.json'), {'command': command, 'config_sha256': sha256(path),
            'source_sha256': sources, 'fresh_initialization': True, 'gpu_uuid': gpu['uuid']})
        state.update(active_condition=name, active_rounds_completed=0, active_run_dir=None)
        status()
        with (root / (name + '.console.log')).open('ab', buffering=0) as log, owned_stage_process(
                command, cwd=source, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL) as child:
            state['active_child_pid'] = child.pid
            status()
            def progress():
                matches = list((root / name).glob('*/round_metrics.json'))
                if len(matches) == 1:
                    try:
                        state.update(active_rounds_completed=len(_read(matches[0])), active_run_dir=str(matches[0].parent))
                    except (OSError, ValueError):
                        pass
                status()
            result = wait_stage_process(child, progress)
        state['active_child_pid'] = None
        status()
        if result:
            raise RuntimeError(name + ' failed with exit ' + str(result))
        matches = list((root / name).glob('*/summary.json'))
        if len(matches) != 1:
            raise ValueError('expected exactly one fresh C stage result')
        run = matches[0].parent
        audit = audit_run(run, rounds, baseline_run, sources, gpu['uuid'], smoke=smoke)
        if smoke:
            if _registry_snapshot(source) != registry_before:
                raise ValueError('C Smoke changed existing checkpoint registry or retained models')
            json_file(root / 'smoke_passed.json', {**audit, 'completed_at': _now()})
        json_file(root / (name + '_audit.json'), audit)
        state.update(active_run_dir=str(run), active_rounds_completed=rounds)
        status()
        return run
    status()
    try:
        with controlled_termination():
            smoke_run, main_run = launch_sequence(launch)
            records = render_results(root, baseline_run, main_run)
            _retention_audit(root)
            check_reference(baseline_run, manifest)
            check_sources_unchanged(sources)
            state.update(status='completed', completed=True, active_condition=None,
                active_child_pid=None, finished_at=_now(), smoke_run_dir=str(smoke_run),
                main_run_dir=str(main_run), runs=records)
            status()
            json_file(root / 'summary.json', state)
            seal_manifest(root, sources, kind='trainer_sender_pipeline_c_vs_existing_v5',
                excluded_files=('supervisor.console.log',))
            return state, root
    except BaseException as error:
        import traceback
        (root / 'exception.log').write_text(traceback.format_exc())
        state.update(status='failed', completed=False, active_child_pid=None,
            error_type=type(error).__name__, error=str(error), finished_at=_now())
        status()
        json_file(root / 'summary.json', state)
        raise

"""v5 ablation orchestration through the existing CLI, never a second trainer."""
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from utils.deit_artifacts import csv_file, gpu_snapshot, json_file
from utils.na_sfl_config import DEFAULTS, HYBRID_METHOD, SD_ONLY_METHOD, load_config
from utils.na_sfl_bn_experiments import (_now, controlled_termination,
    owned_stage_process, wait_stage_process, _retention_audit)
from utils.na_sfl_hybrid_experiment import accuracy_record, model_traffic, validate_rate_rows
from utils.na_sfl_hybrid import sample_indices
from utils.na_sfl_network import rate_scenario
from utils.sflv2_metrics import sha256, snapshot_sources, check_sources_unchanged, seal_manifest


def _read(path):
    return json.loads(Path(path).read_text())


def validate_config(baseline, actual):
    if (baseline['method'] != HYBRID_METHOD or baseline['rounds'] != 100
            or baseline['bn_policy'] != 'dual_running_stats_shared_affine'
            or baseline['random_sd'] != {'sample_ratio': .1, 'sample_seed': 42}
            or baseline['prototype']['fast_sd_weight'] != .8
            or baseline['prototype']['fast_proto_weight'] != .2):
        raise ValueError('SD-only reference must be the completed v5 SD10% Hybrid')
    expected = deepcopy(baseline)
    expected.update(method=SD_ONLY_METHOD, bn_policy='shared_original',
        prototype={**baseline['prototype'], 'fast_sd_weight': 1., 'fast_proto_weight': 0.,
                   'slow_proto_weight': 0., 'alignment_weight_beta': 0.},
        result_path=actual['result_path'], device=actual['device'])
    if expected != actual:
        differences = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
        raise ValueError('SD-only config differs beyond Prototype removal / unit CE / output: '+str(differences))


def launch_sequence(launch):
    smoke = launch('smoke_test', 2, True)
    main = launch('run_100round', 100, False)
    return smoke, main


def audit_run(run, rounds, baseline_run, sources, gpu_uuid, *, smoke=False):
    summary, rows, env = (_read(run/name) for name in ('summary.json', 'round_metrics.json', 'environment.json'))
    baseline = _read(baseline_run/'summary.json')
    if (not summary['completed'] or summary['completed_rounds'] != rounds
            or summary['samples_seen'] != 30000*rounds or summary['server_samples_seen'] != 24610*rounds
            or summary['server_updates'] != 240*rounds or summary['auxiliary_heads'] != 0
            or [r['round'] for r in rows] != list(range(1, rounds+1))):
        raise ValueError('SD-only visits / effective SD / steps / round contract failed')
    if any(summary[k] != baseline[k] for k in ('initial_model_hashes', 'partition_sha256', 'data_sha256')):
        raise ValueError('SD-only initial model / data identity differs from v5')
    if env['gpu']['uuid'] != gpu_uuid or env['source_sha256'] != sources or env['prototype_policy'] is not None:
        raise ValueError('SD-only GPU / source / no-Prototype policy mismatch')
    validate_rate_rows(rows, _read(baseline_run/'round_metrics.json'))
    config = _read(run/'config.json')
    if summary['network_rate_scenario_sha256'] != rate_scenario(config['network'], rounds)['sha256']:
        raise ValueError('SD-only network scenario horizon mismatch')
    references = {}
    for line in (baseline_run/'per_job.jsonl').open():
        row = json.loads(line)
        if row['round'] <= rounds:
            references[tuple(row['job_id'])] = (row['input_sha256'], row['labels'], row['sample_ids'])
    from models.model import model_layout
    import math
    feature_bytes = 4*math.prod(model_layout(config['model'], config['cut'])['boundary_shape_per_sample'])
    jobs = slow_sd = 0
    sums = {key: 0 for key in ('sd_payload_bytes', 'gradient_payload_bytes')}
    for line in (run/'per_job.jsonl').open():
        job = json.loads(line)
        key = tuple(job['job_id'])
        indices = (sample_indices(job['samples'], .1, 42, key).tolist()
                   if job['client_id'] in (0, 5) else list(range(job['samples'])))
        if (references.pop(key, None) != (job['input_sha256'], job['labels'], job['sample_ids'])
                or job['selected_indices'] != indices or job['server_samples'] != len(indices)
                or job['transmitted_labels'] != [job['labels'][i] for i in indices]
                or job['prototype_used'] is not False or job['sd_loss_weight'] != 1.
                or 'job_kind' in job or job['activation_bytes'] != len(indices)*feature_bytes
                or job['gradient_bytes'] != len(indices)*feature_bytes
                or job['sd_payload_bytes'] != len(indices)*(feature_bytes+8)
                or job['gradient_payload_bytes'] != len(indices)*feature_bytes):
            raise ValueError('SD-only input / sample / gradient / float byte contract failed')
        if job['client_id'] in (0, 5):
            slow_sd += len(indices)
        for name in sums:
            sums[name] += job[name]
        jobs += 1
    if references or jobs != 240*rounds or slow_sd != 610*rounds:
        raise ValueError('SD-only jobs or Slow sampled SD count mismatch')
    bn = summary['server_batchnorm']
    if (bn['proto_forward_calls'] or bn['global_prototype_updates'] or len(bn['running_counters']) != 10
            or any(v != 240*rounds for v in bn['running_counters'].values())):
        raise ValueError('SD-only actual Server BN tensor counters mismatch')
    boundaries = _read(run/'execution_boundaries.json')
    if len(boundaries) != rounds:
        raise ValueError('missing SD-only drain boundaries')
    for boundary in boundaries:
        transport = boundary['transport']
        if (not boundary['success'] or not boundary['queue_empty'] or not boundary['streams_completed']
                or any(boundary[k] for k in ('workers_alive', 'pending_graphs', 'unresolved_futures'))
                or transport['frames_sent'] != 480 or transport['frames_received'] != 480
                or any(transport[k] for k in ('workers_alive', 'pending_sd', 'pending_gradients'))):
            raise ValueError('SD-only TCP / actor drain audit failed')
    for name, prefix in (('sd_payload_bytes', 'sd'), ('gradient_payload_bytes', 'gradient')):
        if any(sum(b['transport'][prefix+'_payload_bytes_'+direction] for b in boundaries) != sums[name]
               for direction in ('sent', 'received')):
            raise ValueError('SD-only actual TCP payload totals mismatch')
    network = _read(run/'network/network_metadata.json')
    if not network['closed'] or network['cleanup_errors']:
        raise ValueError('SD-only network did not close cleanly')
    manifest = _read(run/'manifest.json')
    for name, digest in manifest['artifacts'].items():
        if sha256(run/name) != digest:
            raise ValueError('SD-only artifact changed: '+name)
    for name, digest in sources.items():
        if sha256(run/'source_snapshot'/name) != digest:
            raise ValueError('SD-only frozen source changed: '+name)
    if smoke and (summary['checkpoint'].get('retained') is not False or not summary['checkpoint']['finite']):
        raise ValueError('SD-only Smoke checkpoint was not verified/discarded')
    return {'passed': True, 'rounds': rounds, 'jobs': jobs, 'client_visits': 30000*rounds,
        'server_sd': 24610*rounds, 'slow_selected_sd': slow_sd, 'prototype_jobs': 0,
        'server_bn_counters': bn['running_counters'], 'tcp_payload_bytes': sums,
        'run_dir': str(run)}


def communication(run):
    totals = {g: {k: 0 for k in ('jobs', 'client_visits', 'actual_sd', 'upload_bytes',
        'download_bytes', 'upload_float_bytes', 'download_float_bytes')} for g in ('fast', 'slow', 'all')}
    for line in (run/'per_job.jsonl').open():
        row = json.loads(line)
        for group in ('slow' if row['client_id'] in (0, 5) else 'fast', 'all'):
            t = totals[group]
            t['jobs'] += 1
            t['client_visits'] += row['samples']
            t['actual_sd'] += row.get('server_samples', row['samples'] if row.get('job_kind') == 'SD_JOB'
                                      else row.get('selected_sd_count', 0))
            for key, source in (('upload_bytes', 'sd_wire_bytes'), ('download_bytes', 'gradient_wire_bytes'),
                ('upload_float_bytes', 'activation_bytes'), ('download_float_bytes', 'gradient_bytes')):
                t[key] += row.get(key, row[source])
    return totals


def render_results(root, baseline_run, main_run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records, curves, traffic = [], [], []
    for name, run in [('NA-SFLv5', baseline_run), ('v5 without Prototype (SD CE=1)', main_run)]:
        summary, rows = _read(run/'summary.json'), _read(run/'round_metrics.json')
        record = accuracy_record(name, run, summary, rows)
        totals = communication(run)
        record.update(totals['all'])
        record.update(model_traffic(run))
        for direction in ('upload', 'download'):
            record['total_'+direction+'_bytes_including_models'] = record[direction+'_bytes']+record['model_'+direction+'_bytes']
        record['delta_last10_vs_v5_pp'] = float(Decimal(record['last10_mean_exact'])-
            Decimal(records[0]['last10_mean_exact'] if records else record['last10_mean_exact']))
        records.append(record)
        curves.append((name, rows))
        traffic.extend({'name': name, 'client_group': group, **values} for group, values in totals.items())
    json_file(root/'comparison.json', records)
    csv_file(root/'comparison.csv', records)
    csv_file(root/'communication_bytes.csv', traffic)
    for filename, xkey, xlabel, divisor in [('accuracy_by_round', 'round', 'Round', 1),
        ('accuracy_by_elapsed', 'elapsed_wall_s', 'Training through evaluation (hours)', 3600)]:
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        for name, rows in curves:
            ax.plot([r[xkey]/divisor for r in rows], [r['test_accuracy'] for r in rows], label=name)
        ax.axhline(70, color='gray', linestyle=':')
        ax.set(xlabel=xlabel, ylabel='Original-SD test accuracy (%)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        for suffix in ('png', 'pdf'):
            fig.savefig(root/(filename+'.'+suffix), dpi=180)
        plt.close(fig)
    lines = ['# v5 vs SD-only ablation', '',
        'Same GPU1 / seed42 / initialization / partition / variable links / sampled indices.',
        'Prototype removed; SD coefficient also changes0.8→1.0. Single seed with actual FCFS order.', '',
        '| Method | Final | Best | Last10 mean | Delta(pp) | TTA70(s) | Wall(s) | Job UL(bytes) | Job DL(bytes) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"|{r['name']}|{r['final_accuracy']:.2f}|{r['best_accuracy']:.2f}|{r['last10_mean']:.3f}|"
            f"{r['delta_last10_vs_v5_pp']:.3f}|{r['tta_70']}|{r['training_through_evaluation_s']:.3f}|"
            f"{r['upload_bytes']}|{r['download_bytes']}|")
    lines += ['', '![Round](accuracy_by_round.png)', '', '![Wall](accuracy_by_elapsed.png)', '',
        'TTA70 is the first evaluated crossing, not sustained accuracy. Bytes include application framing but exclude TCP/IP/link overhead.',
        'Client full-batch forward / BN coupling and visit-weighted FedAvg are retained; Slow server loss sees only selected SD.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return records


def run_sd_only_experiment(args):
    from utils.na_sfl_run import validate_training_gpu
    config = args.config
    source = Path(__file__).resolve().parents[1]
    baseline_run = Path(args.baseline_reference).resolve()
    baseline = _read(baseline_run/'summary.json')
    baseline_config = load_config(overrides={k: v for k, v in _read(baseline_run/'config.json').items() if k in DEFAULTS})
    validate_config(baseline_config, config)
    if not baseline['completed'] or baseline['completed_rounds'] != 100:
        raise ValueError('v5 baseline is not complete')
    baseline_manifest = _read(baseline_run/'manifest.json')
    def check_reference():
        for name, digest in baseline_manifest['artifacts'].items():
            if sha256(baseline_run/name) != digest:
                raise ValueError('v5 original artifact changed: '+name)
    check_reference()
    gpu = gpu_snapshot(config['device'])
    validate_training_gpu(gpu, allow_unprofiled=config['allow_unprofiled_gpu'])
    if gpu['uuid'] != _read(baseline_run/'environment.json')['gpu']['uuid']:
        raise ValueError('SD-only must use the original v5 GPU UUID')
    if rate_scenario(config['network'], 100)['sha256'] != baseline['network_rate_scenario_sha256']:
        raise ValueError('SD-only network scenario differs from v5')
    root = Path(config['result_path'])
    if (root == baseline_run or root.is_relative_to(baseline_run) or baseline_run.is_relative_to(root)
            or (root/'status.json').exists() or any((root/n).exists() for n in ('smoke_test', 'run_100round'))):
        raise ValueError('SD-only output overlaps an existing launch/reference')
    root.mkdir(parents=True, exist_ok=True)
    sources = snapshot_sources(root)
    json_file(root/'source_sha256.json', sources)
    json_file(root/'baseline_reference.json', {'run_dir': str(baseline_run), 'artifacts': baseline_manifest['artifacts']})
    (root/'inputs').mkdir(exist_ok=True)
    state = {'status': 'running', 'completed': False, 'started_at': _now(), 'supervisor_pid': os.getpid(),
        'gpu_uuid': gpu['uuid'], 'active_condition': None, 'active_child_pid': None, 'active_rounds_completed': 0}
    def status():
        state['updated_at'] = _now()
        json_file(root/'status.json', state)
    def launch(name, rounds, smoke):
        check_sources_unchanged(sources)
        check_reference()
        registry = (source/'results/checkpoint_registry.json').read_bytes()
        stage_config = {**{k: config[k] for k in DEFAULTS}, 'rounds': rounds,
            'smoke_test': smoke, 'result_path': str(root/name)}
        path = root/'inputs'/(name+'.yaml')
        path.write_text(yaml.safe_dump(stage_config, sort_keys=True))
        command = [sys.executable, '-u', str(source/'run_exp_main.py'), '--config', str(path)]
        json_file(root/(name+'_launch.json'), {'command': command, 'config_sha256': sha256(path), 'source_sha256': sources})
        state.update(active_condition=name, active_rounds_completed=0, active_run_dir=None)
        status()
        with (root/(name+'.console.log')).open('ab', buffering=0) as log, owned_stage_process(
                command, cwd=source, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL) as child:
            state['active_child_pid'] = child.pid
            status()
            def progress():
                paths = list((root/name).glob('*/round_metrics.json'))
                if len(paths) == 1:
                    try:
                        state.update(active_rounds_completed=len(_read(paths[0])), active_run_dir=str(paths[0].parent))
                    except (OSError, ValueError):
                        pass
                status()
            result = wait_stage_process(child, progress)
        state['active_child_pid'] = None
        status()
        if result:
            raise RuntimeError(name+' failed with exit '+str(result))
        paths = list((root/name).glob('*/summary.json'))
        if len(paths) != 1:
            raise ValueError('expected exactly one SD-only stage result')
        run = paths[0].parent
        audit = audit_run(run, rounds, baseline_run, sources, gpu['uuid'], smoke=smoke)
        if smoke:
            if (source/'results/checkpoint_registry.json').read_bytes() != registry:
                raise ValueError('Smoke changed retained checkpoint registry')
            json_file(root/'smoke_passed.json', {**audit, 'completed_at': _now()})
        json_file(root/(name+'_audit.json'), audit)
        state.update(active_run_dir=str(run), active_rounds_completed=rounds)
        status()
        return run
    status()
    try:
        with controlled_termination():
            smoke_run, main_run = launch_sequence(launch)
            records = render_results(root, baseline_run, main_run)
            _retention_audit(root)
            check_reference()
            check_sources_unchanged(sources)
            state.update(status='completed', completed=True, active_condition=None, finished_at=_now(),
                smoke_run_dir=str(smoke_run), main_run_dir=str(main_run), runs=records)
            status()
            json_file(root/'summary.json', state)
            seal_manifest(root, sources, kind='v5_no_prototype_ablation')
            return state, root
    except BaseException as error:
        import traceback
        (root/'exception.log').write_text(traceback.format_exc())
        state.update(status='failed', completed=False, active_child_pid=None,
            error_type=type(error).__name__, error=str(error))
        status()
        raise

"""Reporting for the existing v3/v4 runner, without model execution.

Wire bytes include the application's measured framing, labels and counts; they
exclude kernel TCP/IP/link overhead. Idle is training wall minus summed server
model CUDA time, including communication/management time, not GPU utilization.
"""
import csv
import json
from pathlib import Path

import yaml

from utils.na_sfl_artifacts import target_summary
from utils.deit_artifacts import json_file


LOSSES = ('L_SD', 'L_ProtoCE', 'L_Align', 'L_Pseudo', 'L_total')
TIMINGS = ('client_forward_compute_s', 'prototype_compute_s', 'server_prototype_compute_s',
           'upload_wait_or_block_s', 'upload_wire_time_s', 'server_queue_wait_s',
           'server_model_compute_s', 'gradient_download_time_s', 'gradient_wait_s',
           'client_backward_update_s', 'sampling_compute_s', 'sd_serialization_time_s',
           'gradient_serialization_time_s')
BYTE_FIELDS = ('upload_bytes', 'download_bytes', 'upload_payload_bytes', 'download_payload_bytes',
               'upload_header_bytes', 'download_header_bytes', 'activation_bytes',
               'upload_float_bytes', 'download_float_bytes')
IDLE_DEFINITION = ('round_train_s minus summed server_model_compute_s; includes communication, '
                   'queue and management time; not physical GPU utilization')


def _first(row, *keys, default=None):
    return next((row[key] for key in keys if row.get(key) is not None), default)


def _duration(row, end, start, fallback=None):
    if row.get(end) is not None and row.get(start) is not None:
        return row[end] - row[start]
    return row.get(fallback) if fallback else None


def _weighted(rows, key):
    own = [row for row in rows if row.get(key) is not None]
    count = sum(row['samples'] for row in own)
    return sum(row[key] * row['samples'] for row in own) / count if count else None


def _sum(rows, key):
    return sum(row[key] for row in rows if row.get(key) is not None)


def _count(value):
    return len(value) if isinstance(value, (list, tuple, dict)) else int(value or 0)


def _csv(path, rows, minimum_fields):
    fields = list(dict.fromkeys([*minimum_fields, *(key for row in rows for key in row)]))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False)
                if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
    temporary.replace(path)


def _jsonl(path, rows):
    temporary = path.with_suffix('.jsonl.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    temporary.replace(path)


def _job(raw, run_id):
    row = dict(raw)
    job_id = row['job_id']
    row.setdefault('run_id', run_id)
    row.setdefault('round', job_id[0])
    row.setdefault('client_id', job_id[1])
    row.setdefault('batch_id', row.get('local_batch_index', job_id[2]))
    row.setdefault('proto_class_ids', [])
    row.setdefault('proto_counts', [])
    row['sd_count'] = row['samples'] if row['job_kind'] == 'SD_JOB' else row.get('selected_sd_count', 0)
    row['actual_sd_count'] = row['sd_count']
    row['actual_proto_count'] = len(row['proto_class_ids'])
    row.setdefault('sample_ratio', 1. if row['job_kind'] == 'SD_JOB' else 0.)
    row.setdefault('upload_float_bytes', row.get('activation_bytes', 0))
    row.setdefault('download_float_bytes', row.get('gradient_bytes', 0))
    aliases = {'server_arrival_ts': ('sd_receive_complete_s', 'enqueue_s'),
               'server_start_ts': ('server_start_s',), 'server_end_ts': ('server_end_s',),
               'upload_done_ts': ('sd_receive_complete_s',),
               'gradient_recv_done_ts': ('gradient_receive_complete_s', 'gradient_received_s')}
    for name, keys in aliases.items():
        row.setdefault(name, _first(row, *keys))
    row.setdefault('gradient_receive_complete_s', row['gradient_recv_done_ts'])
    for direction, source in (('upload', 'sd'), ('download', 'gradient')):
        payload = _first(row, direction + '_payload_bytes', source + '_payload_bytes', default=0)
        header = _first(row, direction + '_header_bytes', source + '_header_bytes', default=0)
        row[direction + '_payload_bytes'] = payload
        row[direction + '_header_bytes'] = header
        row[direction + '_bytes'] = _first(row, direction + '_bytes', source + '_wire_bytes',
                                           default=payload + header)
    row.setdefault('downlink_payload_bytes', row['download_payload_bytes'])
    row['upload_wait_or_block_s'] = _duration(row, 'sd_send_return_s', 'sd_send_start_s',
                                               'upload_wait_or_block_s')
    row['upload_wire_time_s'] = _duration(row, 'sd_receive_complete_s', 'sd_send_start_s',
                                           'upload_wire_time_s')
    row['server_queue_wait_s'] = (_duration(row, 'dequeue_s', 'enqueue_s', 'queue_wait_s')
        if row.get('dequeue_s') is not None else _duration(row, 'server_start_s', 'enqueue_s', 'queue_wait_s'))
    row['gradient_download_time_s'] = _duration(row, 'gradient_receive_complete_s',
                                                 'gradient_send_start_s', 'gradient_download_time_s')
    row['gradient_wait_s'] = _duration(row, 'gradient_receive_complete_s',
                                        'sd_receive_complete_s', 'gradient_wait_s')
    row['client_backward_update_s'] = _duration(row, 'client_compute_done_s',
                                                  'client_backward_start_s', 'client_backward_update_s')
    return row


def _totals(jobs):
    slow = [job for job in jobs if job['job_kind'] in ('PROTO_JOB', 'HYBRID_JOB')]
    result = {key: _sum(jobs, key) for key in BYTE_FIELDS}
    result.update(total_sd_jobs=sum(job['job_kind'] == 'SD_JOB' for job in jobs),
                  total_proto_jobs=len(slow),
                  total_hybrid_jobs=sum(job['job_kind'] == 'HYBRID_JOB' for job in jobs),
                  actual_sd_count=_sum(jobs, 'actual_sd_count'),
                  actual_proto_count=_sum(jobs, 'actual_proto_count'),
                  avg_classes_per_slow_proto=(sum(len(job['proto_class_ids']) for job in slow)
                                             / len(slow) if slow else None))
    result.update({key: _weighted(jobs, key) for key in LOSSES})
    return result


def _steps(round_rows, jobs):
    """Round increments are authoritative; a per-job forward is not an Adam step."""
    if all(row.get('server_updates') is not None for row in round_rows):
        return sum(row['server_updates'] for row in round_rows)
    identified = {(row['round'], row['server_optimizer_step_id']) for row in jobs
                  if row.get('server_optimizer_step_id') is not None}
    unidentified = sum(row.get('server_optimizer_step_id') is None for row in jobs)
    return len(identified) + unidentified


def _pseudo(jobs):
    return {'pseudo_samples_generated': _sum(jobs, 'pseudo_samples_generated'),
            'pseudo_fallback_classes': sum(_count(row.get('pseudo_fallback_classes')) for row in jobs),
            'pseudo_loss': _weighted(jobs, 'L_Pseudo')}


def _render(root, rows, initial_evaluation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    evaluations = ([{'round': 0, **initial_evaluation}] if initial_evaluation else []) + rows
    specs = (
        ('accuracy_vs_round', evaluations, 'round', [('test_accuracy', 'Test accuracy (%)')]),
        ('accuracy_vs_elapsed', evaluations, 'elapsed_wall_s', [('test_accuracy', 'Test accuracy (%)')]),
        ('loss_vs_round', rows, 'round', [('test_ce_loss', 'Test CE'), *[(key, key) for key in LOSSES]]),
        ('per_round_time', rows, 'round', [('round_time_s', 'Round wall seconds'),
                                        ('round_train_s', 'Training seconds')]),
        ('server_idle', rows, 'round', [('server_idle_pct', 'Server non-model time (%)')]),
        ('communication_bytes', rows, 'round', [('upload_bytes', 'Upload wire bytes'),
                                              ('download_bytes', 'Download wire bytes')]),
    )
    for name, data, xkey, curves in specs:
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for key, label in curves:
            points = [row for row in data if row.get(xkey) is not None and row.get(key) is not None]
            if points:
                axis.plot([row[xkey] for row in points], [row[key] for row in points], marker='.', label=label)
        axis.set_xlabel('Elapsed wall seconds from initial distribution' if xkey == 'elapsed_wall_s'
                        else 'Communication round')
        axis.set_ylabel(curves[0][1] if len(curves) == 1 else 'Loss' if name == 'loss_vs_round'
                        else 'Bytes' if name == 'communication_bytes' else 'Seconds')
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=8)
        fig.savefig(root / (name + '.png'), dpi=140)
        plt.close(fig)


def write_proto_artifacts(run_dir, rounds, jobs, aggregation_records, *, config,
                          initial_evaluation=None, render=False):
    """Rewrite reports from all completed records; return extras for runner summary.

    Missing measurements remain null in timing/loss columns. Byte fields count
    job traffic, excluding model broadcast/FedAvg (runner network records retain
    those separately). Raw job, round, prototype and aggregation fields survive.
    Client timings are per-round sums; group timing fields are per-client means.
    """
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    rounds = list(rounds)
    jobs = [_job(job, root.name) for job in jobs]
    slow_ids = set(config.get('slow_client_ids', []))
    aggregation = {row['round']: row for row in aggregation_records}
    v4 = config.get('method') == 'na_sfl_v4_residual'
    per_round, per_client, prototypes, residuals = [], [], [], []
    for raw in rounds:
        row = dict(raw)
        own = [job for job in jobs if job['round'] == row['round']]
        train = _first(row, 'round_train_s', 'training_wall_s_debug_only')
        row.update(_totals(own))
        row['server_optimizer_steps'] = _steps([raw], own)
        row['round_train_s'] = train
        row['round_time_s'] = _first(raw, 'round_time_s', 'round_wall_s_debug_only')
        row['server_model_compute_s'] = _sum(own, 'server_model_compute_s')
        row['server_idle_s'] = train - row['server_model_compute_s'] if train is not None else None
        row['server_idle_pct'] = 100 * row['server_idle_s'] / train if train else None
        row['aggregation_record'] = aggregation.get(row['round'])
        clients = sorted(set(range(config.get('n_clients', config.get('num_clients', 0)))) | {job['client_id'] for job in own})
        local_rows = []
        for cid in clients:
            client_jobs = [job for job in own if job['client_id'] == cid]
            complete = max((job['client_complete_s'] for job in client_jobs
                            if job.get('client_complete_s') is not None), default=None)
            drain = _first(raw, 'worker_drain_elapsed_s', default=train)
            item = {'round': row['round'], 'client_id': cid,
                    'client_group': 'slow' if cid in slow_ids else 'fast',
                    'sample_ratio': config.get('random_sd', {}).get('sample_ratio', 0.) if cid in slow_ids else 1.,
                    'samples': _sum(client_jobs, 'samples'), 'jobs': len(client_jobs),
                    **_totals(client_jobs),
                    **{key: _sum(client_jobs, key) if any(job.get(key) is not None for job in client_jobs)
                       else None for key in TIMINGS},
                    'client_complete_s': complete,
                    'client_round_end_wait_s': max(0., drain - complete)
                        if drain is not None and complete is not None else None,
                    'round_total_s': row['round_time_s']}
            if v4:
                item.update(_pseudo(client_jobs))
            per_client.append(item)
            local_rows.append(item)
        for group in ('fast', 'slow'):
            for metric in ('gradient_wait_s', 'client_round_end_wait_s'):
                values = [item[metric] for item in local_rows
                          if item['client_group'] == group and item[metric] is not None]
                key = group + '_client_' + ('round_end_wait_s' if metric == 'client_round_end_wait_s' else metric)
                row[key] = sum(values) / len(values) if values else None
        proto = raw.get('global_prototype_stats')
        if proto is not None:
            prototypes.append({'round': row['round'], **proto})
        if v4:
            row.update(_pseudo(own))
            stats = raw.get('residual_bank_stats')
            if stats is None and isinstance(proto, dict):
                stats = proto.get('residual_bank_stats')
            residuals.append({'round': row['round'], **(stats or {}), **_pseudo(own)})
        per_round.append(row)

    _csv(root / 'per_round.csv', per_round, ('round', 'test_accuracy', 'test_ce_loss'))
    _csv(root / 'per_client_round.csv', per_client, ('round', 'client_id', 'client_group'))
    _jsonl(root / 'per_job.jsonl', jobs)
    _jsonl(root / 'global_prototype_stats.jsonl', prototypes)
    if v4:
        _jsonl(root / 'residual_bank_stats.jsonl', residuals)
    temporary = root / 'config_resolved.yaml.tmp'
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    temporary.replace(root / 'config_resolved.yaml')
    partition = root / 'partition.json'
    if partition.exists():
        json_file(root / 'data_partition_summary.json', json.loads(partition.read_text()))
    elif not (root / 'data_partition_summary.json').exists():
        json_file(root / 'data_partition_summary.json', {'status': 'partition_not_available',
            'num_clients': config.get('num_clients'), 'samples_per_client': config.get('samples_per_client'),
            'seed': config.get('seed')})

    evaluated = ([{'round': 0, **initial_evaluation}] if initial_evaluation else []) + rounds
    best = max(evaluated, key=lambda row: row['test_accuracy']) if evaluated else None
    final = rounds[-1] if rounds else None
    summary = {**_totals(jobs), 'server_optimizer_steps': _steps(rounds, jobs),
        'best_accuracy': best['test_accuracy'] if best else None, 'best_round': best['round'] if best else None,
        'final_accuracy': final['test_accuracy'] if final else None,
        'final_round': final['round'] if final else None,
        'targets': [target_summary(rounds, target, initial_evaluation=initial_evaluation)
                    for target in config.get('metrics', {}).get('target_accuracies', [])],
        'server_idle_definition': IDLE_DEFINITION,
        'communication_bytes_scope': 'job traffic including application framing; excludes model/FedAvg and TCP/IP/link headers',
        'loss_aggregation': 'actual input sample weighted mean over jobs with each recorded term',
        'client_group_timing_aggregation': 'mean of per-client round sums'}
    if v4:
        summary.update(_pseudo(jobs))
        summary['pseudo_gradient_to_client'] = config.get('v4', {}).get('pseudo_gradient_to_client')
        for key in ('residual_bank_size_by_class', 'class_donor_coverage'):
            summary[key] = residuals[-1].get(key) if residuals else None
    if render:
        _render(root / 'plots', per_round, initial_evaluation)
    return summary

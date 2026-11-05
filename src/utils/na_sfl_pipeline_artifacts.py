"""CPU-only event reports for C=w2_accum2, without rewriting legacy artifacts.

All clocks are round-relative. Durations from different rounds are summed only
after each round's interval union. Missing measurements stay null, including
GPU utilization: wall service idle is not physical device utilization.
"""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path


BYTE_ALIASES = {
    'upload_bytes': ('upload_bytes', 'sd_wire_bytes'),
    'download_bytes': ('download_bytes', 'gradient_wire_bytes'),
    'upload_payload_bytes': ('upload_payload_bytes', 'sd_payload_bytes'),
    'download_payload_bytes': ('download_payload_bytes', 'gradient_payload_bytes'),
    'upload_float_bytes': ('upload_float_bytes', 'activation_bytes'),
    'download_float_bytes': ('download_float_bytes', 'gradient_bytes'),
}
DEFINITIONS = {
    'server_queue': 'server_start_s - sd_receive_complete_s; includes validation/admission and internal queue',
    'server_internal_queue': 'dequeue_s - enqueue_s; stored separately from receive-to-service waiting',
    'server_queue_depth': 'half-open receive-to-service intervals; area / training_wall_s_debug_only',
    'server_busy_wall': 'union of server_start_s/end_s per round; not GPU utilization',
    'server_cuda_model_busy': 'sum of measured model forward/backward/optimizer CUDA durations, separate from service wall',
    'forward_uplink_overlap': 'per-round/client union of intersections of different jobs: aligned CUDA '
        'Forward events with [sd_send_start_s, sd_receive_complete_s]; explicitly labelled host-phase fallback',
    'forward_uplink_overlap_lower_bound': 'aligned CUDA Forward intervals shrink at both ends by '
        'cuda_event_origin_host_uncertainty_s before their UL intersections are unioned',
    'forward_uplink_overlap_host_phase': '[client_forward_start_s, forward_compute_end_s] vs other-job UL; '
        'includes host scheduling overhead; never labelled CUDA computation',
    'forward_uplink_overlap_paced': 'same union with activation_ready_s as the Forward end; includes pacing',
    'client_time_totals': 'client-seconds summed across clients; simultaneous clients may sum beyond round wall',
    'client_idle_wait': 'actual Future wait charged only to a group optimizer owner, once per group',
    'gradient_pending': 'gradient_available_s - forward_compute_end_s; can overlap useful work',
    'round_end_wait': 'worker_drain_elapsed_s - last group_complete_s for each client',
    'sender_queue_wait': 'sender_dequeue_s - sendqueue_enqueue_s; excludes active transmission',
    'graph_bytes': 'retained boundary tensor byte lower bound, not total autograd or GPU memory',
    'traffic': 'job application wire bytes; excludes model/FedAvg traffic and TCP/IP/link overhead',
    'missing': 'null means not measured or incomplete; a measured zero remains zero',
}


def _number(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('nonfinite or nonnumeric event measurement: ' + name)
    return value


def _first(row, names):
    return next((row[name] for name in names if row.get(name) is not None), None)


def _duration(row, start, end):
    a, b = (_number(row.get(key), key) for key in (start, end))
    if a is None or b is None:
        return None
    if b < a - 1e-9:
        raise ValueError('reversed event interval: ' + start + ' -> ' + end)
    return max(0., b - a)


def _sum(values):
    values = list(values)
    return sum(values) if values and all(value is not None for value in values) else None


def _max(values):
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _peak(rows, *fields):
    return _max(_number(row.get(field), field) for row in rows for field in fields)


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(result[-1][1], end)
        else:
            result.append((start, end))
    return result


def _intervals(rows, start, end):
    if not rows or any(_duration(row, start, end) is None for row in rows):
        return None
    return [(row[start], row[end]) for row in rows]


def _union_time(intervals):
    return sum(end - start for start, end in _union(intervals)) if intervals is not None else None


def _overlap(jobs, forward_end):
    forwards = _intervals(jobs, 'client_forward_start_s', forward_end)
    uploads = _intervals(jobs, 'sd_send_start_s', 'sd_receive_complete_s')
    return _cross_overlap(forwards, uploads)


def _cross_overlap(forwards, uploads):
    if forwards is None or uploads is None:
        return None
    intersections = []
    for i, (start, end) in enumerate(forwards):
        for j, (ul_start, ul_end) in enumerate(uploads):
            if i != j and start < ul_end and ul_start < end:
                intersections.append((max(start, ul_start), min(end, ul_end)))
    return _union_time(intersections)


def _forward_metrics(jobs):
    host = _overlap(jobs, 'forward_compute_end_s')
    aligned = _intervals(jobs, 'aligned_cuda_forward_start_s', 'aligned_cuda_forward_end_s')
    uploads = _intervals(jobs, 'sd_send_start_s', 'sd_receive_complete_s')
    actual = _cross_overlap(aligned, uploads)
    lower = None
    if actual is not None:
        uncertainty = [_number(row.get('cuda_event_origin_host_uncertainty_s'),
                               'cuda_event_origin_host_uncertainty_s') for row in jobs]
        if all(value is not None for value in uncertainty):
            if any(value < 0 for value in uncertainty):
                raise ValueError('negative CUDA event clock alignment uncertainty')
            core = [(start + error, max(start + error, end - error))
                    for (start, end), error in zip(aligned, uncertainty)]
            lower = _cross_overlap(core, uploads)
    return {'client_forward_uplink_overlap_s': actual if actual is not None else host,
        'client_forward_uplink_overlap_lower_bound_s': lower,
        'client_forward_uplink_overlap_host_phase_s': host,
        'client_forward_uplink_overlap_paced_s': _overlap(jobs, 'activation_ready_s'),
        'client_forward_uplink_overlap_clock': 'aligned_cuda_events' if actual is not None
            else 'host_forward_phase' if host is not None else None}


def _clock(rows):
    clocks = {row['client_forward_uplink_overlap_clock'] for row in rows}
    if not clocks or None in clocks:
        return None
    return next(iter(clocks)) if len(clocks) == 1 else 'mixed_aligned_cuda_and_host_forward_phase'


def _stats(values):
    values = list(values)
    if not values or any(value is None for value in values):
        return {'mean_s': None, 'p95_s': None, 'max_s': None}
    values.sort()
    location = .95 * (len(values) - 1)
    lower, upper = math.floor(location), math.ceil(location)
    p95 = values[lower] + (values[upper] - values[lower]) * (location - lower)
    return {'mean_s': sum(values) / len(values), 'p95_s': p95, 'max_s': values[-1]}


def _queue_metrics(jobs, training_wall):
    waits = [row['server_queue_wait_s'] for row in jobs]
    result = {'server_queue_wait_' + key: value for key, value in _stats(waits).items()}
    internal = _stats(row['server_internal_queue_wait_s'] for row in jobs)
    result.update({'server_internal_queue_wait_' + key: value for key, value in internal.items()})
    area = _sum(waits)
    depth = None
    if area is not None:
        events = defaultdict(int)
        for row in jobs:
            if row['server_start_s'] > row['sd_receive_complete_s']:
                events[row['sd_receive_complete_s']] += 1
                events[row['server_start_s']] -= 1
        current = depth = 0
        for stamp in sorted(events):
            current += events[stamp]  # Half-open intervals: ties create no transient extra job.
            depth = max(depth, current)
    result.update(server_queue_area_job_s=area, server_queue_depth_max=depth,
        server_queue_depth_mean=area / training_wall if area is not None and training_wall else None)
    return result


def _group_metrics(jobs):
    if not jobs or any(row.get('pipeline_group_id') is None
                      or type(row.get('client_optimizer_step_performed')) is not bool for row in jobs):
        return {'client_optimizer_steps': None, 'client_idle_wait_s': None,
                'client_optimizer_wall_s': None}
    groups = defaultdict(list)
    for row in jobs:
        groups[row['pipeline_group_id']].append(row)
    steps, waits, intervals = 0, [], []
    for rows in groups.values():
        owners = [row for row in rows if row['client_optimizer_step_performed']]
        if not owners:
            continue
        steps += 1
        # Repeated metadata on both jobs never doubles a group-level measurement.
        waits.append(_max(_number(row.get('client_idle_wait_s'), 'client_idle_wait_s') for row in owners))
        spans = _intervals(owners, 'group_optimizer_step_start_s', 'group_optimizer_step_end_s')
        intervals.append(_union_time(spans))
    return {'client_optimizer_steps': steps, 'client_idle_wait_s': _sum(waits),
            'client_optimizer_wall_s': _sum(intervals)}


def _job(raw, run_id):
    row = dict(raw)
    row.setdefault('run_id', run_id)
    row.setdefault('round', row['job_id'][0])
    row.setdefault('client_id', row['job_id'][1])
    cuda_duration = _duration(row, 'cuda_client_forward_start_s', 'cuda_client_forward_end_s')
    offset = _number(row.get('cuda_event_origin_host_offset_s'), 'cuda_event_origin_host_offset_s')
    for endpoint in ('start', 'end'):
        row['aligned_cuda_forward_' + endpoint + '_s'] = (offset + row['cuda_client_forward_' + endpoint + '_s']
            if cuda_duration is not None and offset is not None else None)
    for key, start, end in (
        ('server_queue_wait_s', 'sd_receive_complete_s', 'server_start_s'),
        ('server_internal_queue_wait_s', 'enqueue_s', 'dequeue_s'),
        ('forward_compute_wall_s', 'client_forward_start_s', 'forward_compute_end_s'),
        ('forward_paced_wall_s', 'client_forward_start_s', 'activation_ready_s'),
        ('uplink_wall_s', 'sd_send_start_s', 'sd_receive_complete_s'),
        ('gradient_pending_s', 'forward_compute_end_s', 'gradient_available_s'),
        ('sender_queue_wait_s', 'sendqueue_enqueue_s', 'sender_dequeue_s')):
        row[key] = _duration(row, start, end)
    for key, aliases in BYTE_ALIASES.items():
        row[key] = _number(_first(row, aliases), key)
    if row.get('server_model_compute_s') is None:
        times = row.get('cuda_time_s') or {}
        row['server_model_compute_s'] = _sum(_number(times.get(key), key)
            for key in ('server_forward', 'server_backward', 'server_optimizer'))
    return row


def _client(raw_round, cid, jobs, slow_ids):
    completions = [_number(row.get('group_complete_s'), 'group_complete_s') for row in jobs]
    completed = max(completions) if completions and all(value is not None for value in completions) else None
    drain = _number(raw_round.get('worker_drain_elapsed_s'), 'worker_drain_elapsed_s')
    return {'round': raw_round['round'], 'client_id': cid,
        'client_group': 'slow' if cid in slow_ids else 'fast',
        'jobs': len(jobs), 'client_visits': sum(row['samples'] for row in jobs),
        **_group_metrics(jobs),
        **_forward_metrics(jobs),
        'gradient_pending_s': _sum(row['gradient_pending_s'] for row in jobs),
        'sender_queue_wait_s': _sum(row['sender_queue_wait_s'] for row in jobs),
        'group_complete_s': completed,
        'round_end_wait_s': max(0., drain - completed) if drain is not None and completed is not None else None,
        'max_inflight': _peak(jobs, 'inflight_on_forward', 'max_outstanding_jobs'),
        'max_pending_graphs': _peak(jobs, 'pending_graph_count', 'max_pending_graphs',
            'inflight_on_forward', 'pending_graph_count_after_backward'),
        'max_pending_graph_boundary_bytes': _peak(jobs, 'pending_graph_bytes'),
        'max_sendqueue_depth': _peak(jobs, 'sendqueue_depth_on_enqueue', 'max_sender_queue_depth'),
        'max_sendqueue_bytes': _peak(jobs, 'sendqueue_cpu_bytes_on_enqueue', 'max_queued_cpu_bytes'),
        'max_completion_queue_depth': _peak(jobs, 'max_completion_queue_depth',
            'completion_queue_depth_on_available', 'completion_queue_depth_after_backward'),
        **{key: _sum(row[key] for row in jobs) for key in BYTE_ALIASES}}


CLIENT_SUMS = ('client_optimizer_steps', 'client_idle_wait_s', 'client_optimizer_wall_s',
    'client_forward_uplink_overlap_s', 'client_forward_uplink_overlap_paced_s',
    'client_forward_uplink_overlap_lower_bound_s', 'client_forward_uplink_overlap_host_phase_s',
    'gradient_pending_s', 'sender_queue_wait_s', 'round_end_wait_s')
PEAKS = ('max_inflight', 'max_pending_graphs', 'max_pending_graph_boundary_bytes',
         'max_sendqueue_depth', 'max_sendqueue_bytes', 'max_completion_queue_depth')


def _round(raw, jobs, clients):
    row = dict(raw)
    train = _number(raw.get('training_wall_s_debug_only'), 'training_wall_s_debug_only')
    if train is not None and train < 0:
        raise ValueError('negative training wall duration')
    busy = _union_time(_intervals(jobs, 'server_start_s', 'server_end_s'))
    row.update(jobs=len(jobs), client_visits=sum(job['samples'] for job in jobs),
        training_wall_s=train, server_busy_wall_s=busy,
        server_idle_wall_s=max(0., train - busy) if train is not None and busy is not None else None,
        server_cuda_model_busy_s=_sum(_number(job.get('server_model_compute_s'), 'server_model_compute_s')
                                     for job in jobs),
        measured_gpu_utilization_percent=None, **_queue_metrics(jobs, train))
    row.update({key: _sum(client[key] for client in clients) for key in CLIENT_SUMS})
    row['client_forward_uplink_overlap_clock'] = _clock(clients)
    row.update({key: _max(client[key] for client in clients) for key in PEAKS})
    row.update({key: _sum(job[key] for job in jobs) for key in BYTE_ALIASES})
    row['server_busy_wall_percent'] = 100 * busy / train if busy is not None and train else None
    row['server_idle_wall_percent'] = 100 * row['server_idle_wall_s'] / train if row['server_idle_wall_s'] is not None and train else None
    return row


def _csv(path, rows):
    fields = list(dict.fromkeys(['round', *(key for row in rows for key in row)]))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False)
                if isinstance(value, (list, tuple, dict)) else value for key, value in row.items()})
    temporary.replace(path)


def _json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _render(root, rounds, jobs, initial_evaluation, slow_ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def save(fig, stem):
        for suffix in ('.png', '.pdf'):
            fig.savefig(root / (stem + suffix), dpi=150)
        plt.close(fig)

    evaluations = ([{'round': 0, **initial_evaluation}] if initial_evaluation else []) + rounds
    for stem, xkey, xlabel in [('pipeline_accuracy_by_round', 'round', 'Round'),
        ('pipeline_accuracy_by_elapsed', 'elapsed_wall_s', 'Training through evaluation (seconds)')]:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        points = [row for row in evaluations if row.get(xkey) is not None and row.get('test_accuracy') is not None]
        ax.plot([row[xkey] for row in points], [row['test_accuracy'] for row in points], marker='.')
        ax.set(xlabel=xlabel, ylabel='Original-SD test accuracy (%)')
        ax.grid(alpha=.2)
        save(fig, stem)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key, label in [('server_queue_wait_mean_s', 'Receive-to-service mean'),
                       ('server_queue_wait_p95_s', 'Receive-to-service P95')]:
        points = [row for row in rounds if row.get(key) is not None]
        if points:
            axes[0].plot([row['round'] for row in points], [row[key] for row in points], label=label)
    for key, label in [('server_idle_wall_s', 'Server service-wall idle'),
                       ('client_idle_wait_s', 'Trainer Future wait (client-seconds)')]:
        points = [row for row in rounds if row.get(key) is not None]
        if points:
            axes[1].plot([row['round'] for row in points], [row[key] for row in points], label=label)
    for ax in axes:
        ax.set(xlabel='Round', ylabel='Seconds')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    axes[1].set_title('Event timings; not measured GPU utilization')
    save(fig, 'pipeline_queue_idle')
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    first_round = min((row['round'] for row in jobs), default=None)
    clients = sorted({row['client_id'] for row in jobs})
    selected = sorted(set(clients[:1] + [cid for cid in clients if cid in slow_ids][:2]
                          + [cid for cid in clients if cid not in slow_ids][:1]))
    ticks, labels = [], []
    for position, cid in enumerate(selected):
        own = sorted((row for row in jobs if row['round'] == first_round and row['client_id'] == cid),
                     key=lambda row: row['job_id'][2])[:4]
        aligned = own and all(row['aligned_cuda_forward_start_s'] is not None for row in own)
        forward_start, forward_end, forward_label = (('aligned_cuda_forward_start_s', 'aligned_cuda_forward_end_s', 'Aligned CUDA Forward')
            if aligned else ('client_forward_start_s', 'forward_compute_end_s', 'Host Forward phase'))
        for offset, start, end, color, label in [(0, forward_start, forward_end, '#0072b2', forward_label),
            (1, 'sd_send_start_s', 'sd_receive_complete_s', '#e69f00', 'UL through full receipt')]:
            y = position * 3 + offset
            for index, row in enumerate(own):
                if _duration(row, start, end) is not None:
                    ax.barh(y, row[end] - row[start], left=row[start], height=.7,
                            color=color, label=label if position == 0 and index == 0 else None)
            ticks.append(y)
            labels.append(f'Client{cid} ' + ('F' if offset == 0 else 'UL'))
    ax.set(yticks=ticks, yticklabels=labels, xlabel='Round-relative seconds',
           title=f'Round{first_round}: first four jobs per shown client; Forward excludes pacing')
    ax.grid(axis='x', alpha=.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    save(fig, 'pipeline_overlap_timeline')


def write_pipeline_artifacts(run_dir, rounds, jobs, *, config, initial_evaluation=None, render=False):
    """Write pipeline-only event artifacts and return JSONable summary extras.

    Inputs and legacy per_job/per_round artifacts are never modified. Server
    queue P95 uses linear interpolation; depth uses half-open event intervals.
    Overlap/idle totals use client-seconds, while Server times use wall seconds.
    """
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [_job(raw, root.name) for raw in jobs]
    raw_rounds = list(rounds)
    slow_ids = set(config.get('slow_client_ids', []))
    round_ids = {raw['round'] for raw in raw_rounds}
    if len(round_ids) != len(raw_rounds) or any(job['round'] not in round_ids for job in jobs):
        raise ValueError('pipeline events must belong to unique recorded rounds')
    identities = [tuple(row['job_id']) for row in jobs]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate pipeline job identity')
    per_round, per_client = [], []
    for raw in raw_rounds:
        own = [row for row in jobs if row['round'] == raw['round']]
        ids = sorted({row['client_id'] for row in own})
        clients = [_client(raw, cid, [row for row in own if row['client_id'] == cid], slow_ids) for cid in ids]
        per_client.extend(clients)
        per_round.append(_round(raw, own, clients))
    summary = {'mode': 'A_w1_step1' if config.get('method') == 'na_sfl_proto_random_sd' else 'w2_accum2',
        'rounds': len(per_round), 'jobs': len(jobs),
        'client_visits': sum(row['samples'] for row in jobs),
        'server_updates': _sum(_number(row.get('server_updates'), 'server_updates') for row in raw_rounds),
        'measured_gpu_utilization_percent': None, 'definitions': dict(DEFINITIONS),
        'round_metrics': per_round}
    sums = (*CLIENT_SUMS, *BYTE_ALIASES, 'training_wall_s', 'server_busy_wall_s',
            'server_idle_wall_s', 'server_cuda_model_busy_s', 'server_queue_area_job_s')
    summary.update({key: _sum(row[key] for row in per_round) for key in sums})
    summary['client_forward_uplink_overlap_clock'] = _clock(per_round)
    summary.update({key: _max(row[key] for row in per_round) for key in PEAKS})
    summary.update({key: _peak(raw_rounds, key) for key in
                    ('gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes')})
    stats = _stats(row['server_queue_wait_s'] for row in jobs)
    summary.update({'server_queue_wait_' + key: value for key, value in stats.items()})
    internal = _stats(row['server_internal_queue_wait_s'] for row in jobs)
    summary.update({'server_internal_queue_wait_' + key: value for key, value in internal.items()})
    summary['server_queue_depth_max'] = _max(row['server_queue_depth_max'] for row in per_round)
    train, area, busy, idle = (summary[key] for key in ('training_wall_s', 'server_queue_area_job_s',
                                                     'server_busy_wall_s', 'server_idle_wall_s'))
    summary['server_queue_depth_mean'] = area / train if area is not None and train else None
    summary['server_busy_wall_percent'] = 100 * busy / train if busy is not None and train else None
    summary['server_idle_wall_percent'] = 100 * idle / train if idle is not None and train else None
    temporary = root / 'pipeline_jobs.jsonl.tmp'
    with temporary.open('w') as stream:
        for row in jobs:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    temporary.replace(root / 'pipeline_jobs.jsonl')
    _csv(root / 'pipeline_events_by_job.csv', jobs)
    _csv(root / 'pipeline_events_by_client.csv', per_client)
    _csv(root / 'pipeline_rounds.csv', per_round)
    extras = {'pipeline': summary}
    _json(root / 'pipeline_summary.json', extras)
    if render:
        _render(root, per_round, jobs, initial_evaluation, slow_ids)
    return extras

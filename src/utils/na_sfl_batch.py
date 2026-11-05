"""Small FIFO batch selection and Client-only time-padding primitives."""
import time


def take_ready_batch(queue, max_samples, policy):
    """Caller owns the queue lock; consume only its currently ready prefix."""
    if policy not in ('individual', 'concat') or type(max_samples) is not int or max_samples < 1:
        raise ValueError('invalid server batch policy or capacity')
    selected, samples = [], 0
    while queue:
        size = queue[0]['row'].get('server_samples', queue[0]['row']['samples'])
        if type(size) is not int or size < 1 or size > max_samples:
            raise ValueError('one SD exceeds the server sample capacity')
        if samples + size > max_samples:
            break
        selected.append(queue.popleft())
        samples += size
        if policy == 'individual':
            break
    return selected


def phase_targets(config, samples, mode):
    targets = {key: float(config.get('client_compute_seconds', {}).get(key, 0.)) * samples / config['batch_size']
               for key in ('forward', 'backward', 'auxiliary')}
    if mode == 'aux':
        targets['auxiliary'] += targets['backward']
        targets['backward'] = 0.
    return targets


def validate_server_batches(jobs, batches, max_samples, policy):
    """Verify actual SD membership and one persistent version step per batch."""
    offset = 0
    for index, batch in enumerate(batches):
        count = batch['sd_count']
        own = jobs[offset:offset+count]
        if (type(count) is not int or count < 1 or len(own) != count
                or (policy == 'individual' and count != 1)
                or batch['samples'] != sum(row.get('server_samples', row['samples']) for row in own)
                or not 0 < batch['samples'] <= max_samples
                or [tuple(row['job_id']) for row in own] != [tuple(j) for j in batch['job_ids']]
                or batch['batch_id'] != batch['server_version_before']
                or batch['server_version_after'] != batch['server_version_before'] + 1
                or (index and batch['server_version_before'] != batches[index-1]['server_version_after'])
                or any(row['round'] != batch['round'] or row['server_batch_id'] != batch['batch_id']
                       or row['server_version_before'] != batch['server_version_before']
                       or row['server_version_after'] != batch['server_version_after'] for row in own)):
            raise ValueError('invalid server batch membership, capacity or optimizer versions')
        offset += count
    if offset != len(jobs):
        raise ValueError('SD not covered exactly once by Server batches')
    return len(batches)


def pace_phase(started, target, aborted, *, clock=time.perf_counter):
    """GPU/Event synchronization belongs to the caller's own Client stream."""
    if aborted.is_set():
        raise RuntimeError('Client pacing aborted')
    elapsed = max(0., clock() - started)
    remaining = max(0., target - elapsed)
    if remaining and aborted.wait(remaining):
        raise RuntimeError('Client pacing aborted')
    return {'target_s': target, 'elapsed_before_padding_s': elapsed,
            'padding_s': max(0., clock() - started - elapsed),
            'overrun_s': max(0., elapsed - target) if target else 0.}

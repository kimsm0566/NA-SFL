"""Live client actors and a single FCFS server on one shared CUDA device.

The caller holds ``sflv2_fp32()`` for the complete worker lifetime. CUDA spans
cover outer node calls, including their validation/copy work; they are elapsed
intervals under shared-GPU contention, not independent-device compute costs.
Only completed, quiescent rounds append jobs to the owner's committed records.
"""

from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeout
import hashlib
import json
import math
import threading
import time
from itertools import islice

import torch
from utils.na_sfl_batch import take_ready_batch, pace_phase, phase_targets


def _map_tensors(value, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, transform) for key, item in value.items()}
    return value


def _float_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() if value.is_floating_point() else 0
    return sum(_float_bytes(item) for item in value.values()) if isinstance(value, dict) else 0


def _stage_cpu(value):
    def stage(tensor):
        result = torch.empty_like(tensor, device='cpu', pin_memory=True)
        result.copy_(tensor, non_blocking=True)
        return result
    return _map_tensors(value, stage)


def async_trace_sha256(rows):
    """Hash the complete, JSON-serializable server-ordered round trace."""
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class _Aborted(Exception):
    pass


class _PipelineCompletions:
    """Completed Futures not yet consumed by a successful Client backward.

    The actor's registry condition protects all calls. This is distinct from
    the transport queue of scheduled receive operations.
    """
    def __init__(self, clients):
        self.pending = set()
        self.depth = [0] * clients
        self.peaks = [0] * clients

    def available(self, job_id):
        job_id = tuple(job_id)
        cid = job_id[1]
        if job_id in self.pending:
            raise RuntimeError('duplicate pipeline gradient completion')
        if self.depth[cid] >= 2:
            raise RuntimeError('pipeline completion queue is full (2)')
        self.pending.add(job_id)
        self.depth[cid] += 1
        self.peaks[cid] = max(self.peaks[cid], self.depth[cid])
        return self.depth[cid]

    def consumed(self, job_id):
        job_id = tuple(job_id)
        if job_id not in self.pending:
            raise RuntimeError('pipeline gradient completion is unavailable')
        self.pending.remove(job_id)
        self.depth[job_id[1]] -= 1
        return self.depth[job_id[1]]

    def clear(self):
        self.pending.clear()
        self.depth[:] = [0] * len(self.depth)


def _pipeline_forward_span(begun, device, outer_span):
    pair = begun.get('forward_cuda_events')
    if isinstance(pair, tuple) and len(pair) == 2 and all(event is not None for event in pair):
        return pair, 'model_forward'
    if device.type == 'cuda':
        raise RuntimeError('pipeline requires exact model-forward CUDA events')
    return outer_span, 'cpu_test_adapter'


def _run_pipeline_groups(node, batches, round_id, cid, forward, backward,
                         step, check_abort, wait_ready):
    """Bound the actor to two graphs and finish a group before any next forward.

    Forward submits immediately, so the first Sender can run during forward2.
    Callbacks retain CUDA ownership in the calling Trainer thread.
    """
    batches = iter(batches)
    while True:
        group = list(islice(batches, 2))
        if not group:
            return
        check_abort()
        node.begin_group([(round_id, cid, batch[0]) for batch in group])
        records = [forward(batch, (round_id, cid, batch[0]), len(group)) for batch in group]
        remaining = list(records)
        while remaining:
            check_abort()
            ready = wait_ready(remaining)
            for record in ready:
                if not any(record is value for value in remaining):
                    raise RuntimeError('duplicate or foreign pipeline completion')
                remaining = [value for value in remaining if value is not record]
                backward(record, not remaining)
        step(records)


def run_async_round(owner, images, labels, partitions, round_id):
    """Run NA-SFL Client/Server workers and commit after full CUDA cleanup.

    Admission follows actual activation readiness. worker_timeout_s detects a
    lack of cooperative progress; it does not limit the total round duration.
    """
    count = len(owner.clients)
    args = owner.args
    pipeline = args.get('method') == 'na_sfl_trainer_sender_pipeline'
    device = torch.device(owner.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("live_async requires CUDA; CPU fallback is forbidden")
    if not count or len(partitions) != count or any(not len(p) for p in partitions):
        raise ValueError("live_async requires one nonempty partition per client")
    if args.get("augmentation", False) and not getattr(owner, 'isolated_augmentation', False):
        raise ValueError("live_async augmentation requires isolated per-client RNG")
    if any(client.pending is not None for client in owner.clients):
        raise RuntimeError("live_async round cannot start with pending client graphs")
    timeout = args.get("worker_timeout_s", 120.)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("worker_timeout_s must be positive and finite")
    timeout = float(timeout)
    wall_origin = time.perf_counter()
    condition = threading.Condition()
    aborted = threading.Event()
    start_barrier = threading.Barrier(count + 2)
    queue, registry, completed_rows, failures = deque(), {}, [], []
    completions = _PipelineCompletions(count) if pipeline else None
    pipeline_pending_rows = {}
    counters = {"admitted_jobs": 0, "serviced_jobs": 0, "completed_jobs": 0,
                "max_queue_length": 0, "serviced_batches": 0}
    expected_jobs = sum(min(math.ceil(len(part) / args['batch_size']) * args['local_epochs'],
                            args.get('max_batches_per_client') or math.inf) for part in partitions)
    clients_done = [0]
    last_progress = [wall_origin]
    stream_events = {}
    server_batches = []
    transport = None
    threads, started_threads = [], []
    client_streams = [torch.cuda.Stream(device=device) for _ in owner.clients]
    server_stream = torch.cuda.Stream(device=device)
    main_stream = torch.cuda.current_stream(device)
    origin = torch.cuda.Event(enable_timing=True)
    alignment_before = time.perf_counter() - wall_origin if pipeline else None
    origin.record(main_stream)
    cuda_origin_offset = cuda_origin_uncertainty = None
    if pipeline:
        origin.synchronize()
        alignment_after = time.perf_counter() - wall_origin
        cuda_origin_offset = (alignment_before + alignment_after) / 2.
        cuda_origin_uncertainty = alignment_after - alignment_before
    main_ready = torch.cuda.Event()
    main_ready.record(main_stream)

    def now():
        return time.perf_counter() - wall_origin

    def progress():
        with condition:
            last_progress[0] = time.perf_counter()

    def cancel(error):
        with condition:
            first_failure = not failures
            if not failures:
                failures.append(error)
            aborted.set()
            if pipeline:
                completions.clear()
                pipeline_pending_rows.clear()
            for future in registry.values():
                if not future.done():
                    future.set_exception(failures[0])
            queue.clear()
            condition.notify_all()
        start_barrier.abort()
        if transport is not None and first_failure:
            transport.abort(error)

    def check_abort():
        if aborted.is_set():
            raise _Aborted()

    def event():
        value = torch.cuda.Event(enable_timing=True)
        value.record()
        return value

    def admit(packet):
        row = packet['row']
        with condition:
            while args.get('queue_capacity') is not None and len(queue) >= args['queue_capacity']:
                check_abort()
                condition.wait(.05)
            check_abort()
            row['arrival_seq'] = counters['admitted_jobs']
            row['enqueue_s'] = now()
            queue.append(packet)
            counters['admitted_jobs'] += 1
            counters['max_queue_length'] = max(counters['max_queue_length'], len(queue))
            row['queue_length_on_enqueue'] = len(queue)
            last_progress[0] = time.perf_counter()
            condition.notify_all()

    def receive_gradient(job_id, gradient):
        with condition:
            check_abort()
            future = registry[tuple(job_id)]
            if future.done():
                raise RuntimeError('duplicate gradient delivery')
            if pipeline:
                row = pipeline_pending_rows[tuple(job_id)]
                row['completion_queue_depth_on_available'] = completions.available(job_id)
                row['max_completion_queue_depth'] = completions.peaks[job_id[1]]
            future.set_result(({'cut_gradient': gradient}, None))
            last_progress[0] = time.perf_counter()
            condition.notify_all()

    def client_work(cid, stream):
        node = owner.clients[cid]
        for index, (batch_images, batch_labels, ids) in owner._client_batches(
                images, labels, partitions[cid], round_id, cid):
            check_abort()
            progress()
            job_id = (round_id, cid, index)
            sample_ids = ids.tolist()
            label_list = batch_labels.tolist()
            credit_started = now()
            if transport is not None:
                transport.reserve(cid, check_abort)
            credit_done = now()
            digest = hashlib.sha256(batch_images.contiguous().numpy().tobytes()).hexdigest()
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            forward_started = now()
            forward_start = event()
            begun = node.begin(job_id, batch_images)
            ready = event()  # Includes the detached boundary clone and label copy.
            ready.synchronize()
            forward_done = ready
            gradient_mode = getattr(node, 'gradient_mode', 'server')
            targets = phase_targets(args, len(sample_ids), gradient_mode)
            forward_padding = pace_phase(wall_origin + forward_started, targets['forward'], aborted)
            activation_ready = now()
            upload = None
            sd_only_upload = None
            if hasattr(node, 'prepare_upload'):
                upload = node.prepare_upload(job_id, begun, batch_labels)
                ready = event()
                ready.synchronize()
            if hasattr(node, 'prepare_sd_upload'):
                sd_only_upload = node.prepare_sd_upload(job_id, begun, batch_labels)
                ready = event()
                ready.synchronize()
            check_abort()
            progress()
            future = Future() if gradient_mode == 'server' else None
            row = {
                "job_id": job_id, "round": round_id, "client_id": cid,
                "round_wall_origin_perf_counter_s":wall_origin,
                "client_compute_model": "shared_gpu_actual_with_client_phase_targets",
                "local_batch_index": index, "sample_ids": sample_ids, "labels": label_list,
                "input_sha256": digest, "samples": len(sample_ids),
                "client_version_before": begun["client_version"],
                "activation_bytes": begun["activation"].numel() * begun["activation"].element_size(),
                "client_forward_start_s": forward_started, "activation_ready_s": activation_ready,
                "client_phase_padding": {'forward': forward_padding},
                "send_buffer_wait_s": credit_done - credit_started,
                "client_thread_ident": threading.get_ident(), "client_stream_id": int(stream.cuda_stream),
                "cuda_time_s": {"client_forward": begun["forward_cuda_s"]},
                "simulated_time_s": None,
                "auxiliary_updated": False, "aux_loss": None,
                "gradient_mode": gradient_mode,
                "_cuda_spans": {"client_forward": (forward_start, forward_done)},
            }
            packet = {"row": row, "activation": begun["activation"], "labels": batch_labels,
                      "ready": ready, "future": future, "return_cut_gradient": gradient_mode == 'server'}
            if upload is not None:
                packet.update({key: upload[key] for key in ('activation', 'labels', 'job_kind', 'counts',
                    'selected_sd', 'selected_labels', 'selected_indices', 'sample_ratio', 'sample_seed') if key in upload})
                row.update(run_id=owner.run_dir.name, job_kind=upload['job_kind'],
                    prototype_compute_s=upload['prototype_compute_s'],
                    client_forward_compute_s=begun['forward_cuda_s'],
                    activation_bytes=_float_bytes(packet['activation']) + _float_bytes(packet.get('selected_sd')))
                if upload['job_kind'] == 'HYBRID_JOB':
                    row.update(sample_ratio=upload['sample_ratio'], sample_seed=upload['sample_seed'],
                        selected_sd_count=upload['selected_sd'].shape[0],
                        selected_indices=upload['selected_indices'].tolist(),
                        sampling_compute_s=upload['sampling_compute_s'])
                packet['actual_batch_size'] = len(sample_ids)
            if sd_only_upload is not None:
                packet.update(activation=sd_only_upload['activation'], labels=sd_only_upload['labels'])
                row.update(server_samples=packet['activation'].shape[0], prototype_used=False,
                    selected_indices=sd_only_upload['selected_indices'].tolist(),
                    transmitted_labels=sd_only_upload['labels'].tolist(),
                    sampling_compute_s=sd_only_upload['sampling_compute_s'], sd_loss_weight=1.,
                    activation_bytes=_float_bytes(packet['activation']))
            with condition:
                if future is not None:
                    registry[job_id] = future
            if transport is None:
                admit(packet)
            else:
                row['sd_host_stage_start_s'] = now()
                for key in ('activation', 'labels', 'counts', 'selected_sd', 'selected_labels', 'selected_indices'):
                    if key in packet:
                        packet[key] = _stage_cpu(packet[key])
                staged = event()
                staged.synchronize()
                row['sd_host_stage_end_s'] = now()
                packet['ready'] = None
                transport.submit(packet)
            if hasattr(node, 'auxiliary_step'):
                row['gradient_mode'] = gradient_mode
                row['auxiliary_start_s'] = now()
                auxiliary_start = event()
                auxiliary = node.auxiliary_step(job_id, batch_labels)
                auxiliary_end = event()
                auxiliary_end.synchronize()
                row['client_phase_padding']['auxiliary'] = pace_phase(
                    wall_origin + row['auxiliary_start_s'], targets['auxiliary'] if auxiliary is not None else 0., aborted)
                row['auxiliary_end_s'] = now()
                row['auxiliary_updated'] = auxiliary is not None
                row['aux_loss'] = auxiliary['loss'] if auxiliary is not None else None
                row['cuda_time_s']['auxiliary_update'] = auxiliary['cuda_s'] if auxiliary is not None else 0.
                row['_cuda_spans']['auxiliary'] = auxiliary_start, auxiliary_end
                if gradient_mode == 'aux':
                    row.update(client_backward_start_s=row['auxiliary_start_s'],
                        client_compute_done_s=now(), client_complete_s=now(),
                        client_version_after=auxiliary['version_after'],
                        gradient_received_s=None, client_gradient_wait_s=None)
                    # The combined auxiliary event includes head/client backward
                    # and both steps. Do not invent individual phase durations.
                    row['cuda_time_s'].update(client_backward=None, client_optimizer=None)
                    with condition:
                        check_abort()
                        counters['completed_jobs'] += 1
                        completed_rows.append(row)
                        last_progress[0] = time.perf_counter()
                        condition.notify_all()
                    del packet, begun, batch_images, batch_labels, row
                    continue
            del packet, begun, batch_images, batch_labels
            wait_started = now()
            while True:
                check_abort()
                try:
                    processed, gradient_ready = future.result(timeout=.05)
                    break
                except FutureTimeout:
                    continue
            row['client_blocked_on_gradient_s'] = now() - wait_started
            if gradient_ready is not None:
                stream.wait_event(gradient_ready)
                gradient_ready.synchronize()
            else:
                processed['cut_gradient'] = _map_tensors(processed['cut_gradient'],
                    lambda tensor: tensor.to(device, non_blocking=True))
            _map_tensors(processed['cut_gradient'], lambda tensor: tensor.record_stream(stream))
            row["gradient_received_s"] = row.get('gradient_receive_complete_s', now())
            row["client_gradient_wait_s"] = row["gradient_received_s"] - row["enqueue_s"]
            check_abort()
            progress()
            row["client_backward_start_s"] = now()
            backward_start = event()
            result = node.complete(job_id, processed["cut_gradient"])
            backward_end = event()
            backward_end.synchronize()
            row['client_phase_padding']['backward'] = pace_phase(
                wall_origin + row['client_backward_start_s'], targets['backward'], aborted)
            row['client_compute_done_s'] = now()
            if upload is not None:
                row['client_backward_update_s'] = row['client_compute_done_s'] - row['client_backward_start_s']
                row['gradient_wait_s'] = row['gradient_received_s'] - row['sd_receive_complete_s'] if transport else row['client_gradient_wait_s']
            row["client_complete_s"] = now()
            row["client_version_after"] = result["version_after"]
            row["cuda_time_s"].update(client_backward=result["backward_cuda_s"],
                                      client_optimizer=result["optimizer_cuda_s"])
            row["_cuda_spans"]["client_backward"] = backward_start, backward_end
            with condition:
                check_abort()
                counters["completed_jobs"] += 1
                completed_rows.append(row)
                registry.pop(job_id)
                last_progress[0] = time.perf_counter()
                condition.notify_all()
            del processed, future, row

    def pipeline_client_work(cid, stream):
        node = owner.clients[cid]
        group_idle = [0.]
        deferred_padding = [None]

        def forward(batch, job_id, group_size):
            index, (batch_images, batch_labels, ids) = batch
            check_abort()
            progress()
            data_ready = now()
            credit_started = now()
            if transport is not None:
                transport.reserve(cid, check_abort)
            credit_done = now()
            sample_ids, label_list = ids.tolist(), batch_labels.tolist()
            digest = hashlib.sha256(batch_images.contiguous().numpy().tobytes()).hexdigest()
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            inflight = node.pending_graph_count + 1
            started = now()
            start = event()
            begun = node.begin(job_id, batch_images)
            end = event()
            end.synchronize()
            forward_span, forward_scope = _pipeline_forward_span(begun, node.device, (start, end))
            compute_end = now()
            targets = phase_targets(args, len(sample_ids), 'server')
            padding = pace_phase(wall_origin + started, targets['forward'], aborted)
            activation_ready = now()
            upload = node.prepare_upload(job_id, begun, batch_labels)
            ready = event()
            ready.synchronize()
            future = Future()
            row = {
                'job_id': job_id, 'round': round_id, 'client_id': cid,
                'round_wall_origin_perf_counter_s': wall_origin,
                'client_compute_model': 'shared_gpu_actual_with_client_phase_targets',
                'local_batch_index': index, 'sample_ids': sample_ids, 'labels': label_list,
                'input_sha256': digest, 'samples': len(sample_ids),
                'client_version_before': begun['client_version'],
                'client_forward_start_s': started, 'forward_compute_end_s': compute_end,
                'activation_ready_s': activation_ready, 'data_ready_s': data_ready,
                'client_phase_padding': {'forward': padding},
                'send_buffer_wait_s': credit_done - credit_started,
                'client_thread_ident': threading.get_ident(), 'client_stream_id': int(stream.cuda_stream),
                'cuda_time_s': {'client_forward': begun['forward_cuda_s']},
                'simulated_time_s': None, 'auxiliary_updated': False, 'aux_loss': None,
                'gradient_mode': 'server', '_cuda_spans': {'client_forward': forward_span},
                'forward_cuda_event_scope': forward_scope,
                'cuda_event_origin_host_offset_s': cuda_origin_offset,
                'cuda_event_origin_host_uncertainty_s': cuda_origin_uncertainty,
                'pipeline_group_id': index // 2, 'group_size': group_size,
                'client_group_version': begun['client_version'],
                'client_optimizer_step_id': begun['client_version'],
                'client_optimizer_step_performed': False,
                'gradient_scale_to_group_mean': 1. / group_size,
                'inflight_on_forward': inflight, 'pending_graph_count': node.pending_graph_count,
                'pending_graph_bytes': node.pending_graph_bytes,
                'max_pending_graphs': node.max_pending_graphs,
                'client_idle_wait_s': 0., 'run_id': owner.run_dir.name,
                'job_kind': upload['job_kind'], 'prototype_compute_s': upload['prototype_compute_s'],
                'client_forward_compute_s': begun['forward_cuda_s'],
            }
            packet = {'row': row, 'ready': ready, 'future': future, 'return_cut_gradient': True,
                      'actual_batch_size': len(sample_ids)}
            packet.update({key: upload[key] for key in ('activation', 'labels', 'job_kind', 'counts',
                'selected_sd', 'selected_labels', 'selected_indices', 'sample_ratio', 'sample_seed') if key in upload})
            row['activation_bytes'] = _float_bytes(packet['activation']) + _float_bytes(packet.get('selected_sd'))
            if upload['job_kind'] == 'HYBRID_JOB':
                row.update(sample_ratio=upload['sample_ratio'], sample_seed=upload['sample_seed'],
                    selected_sd_count=upload['selected_sd'].shape[0],
                    selected_indices=upload['selected_indices'].tolist(),
                    sampling_compute_s=upload['sampling_compute_s'])
            with condition:
                check_abort()
                if job_id in registry:
                    raise RuntimeError('duplicate pipeline Future')
                registry[job_id] = future
                pipeline_pending_rows[job_id] = row
            if transport is None:
                admit(packet)
            else:
                row['sd_host_stage_start_s'] = now()
                for key in ('activation', 'labels', 'counts', 'selected_sd', 'selected_labels', 'selected_indices'):
                    if key in packet:
                        packet[key] = _stage_cpu(packet[key])
                staged = event()
                staged.synchronize()
                row['sd_host_stage_end_s'] = now()
                packet['ready'] = None
                transport.submit(packet)
            progress()
            return {'row': row, 'future': future, 'targets': targets}

        def wait_ready(records):
            with condition:
                waiting = not any(record['future'].done() for record in records)
                started = now()
                while not any(record['future'].done() for record in records):
                    check_abort()
                    condition.wait(.05)
                if waiting:
                    group_idle[0] += now() - started
                return [record for record in records if record['future'].done()]

        def backward(record, last):
            row = record['row']
            processed, ready = record['future'].result()
            if ready is not None:
                stream.wait_event(ready)
                ready.synchronize()
            else:
                processed['cut_gradient'] = _map_tensors(processed['cut_gradient'],
                    lambda tensor: tensor.to(device, non_blocking=True))
            _map_tensors(processed['cut_gradient'], lambda tensor: tensor.record_stream(stream))
            row['gradient_available_s'] = row.get('gradient_available_s', now())
            row['gradient_received_s'] = row.get('gradient_receive_complete_s', row['gradient_available_s'])
            row['client_gradient_wait_s'] = row['gradient_received_s'] - row['enqueue_s']
            row['client_blocked_on_gradient_s'] = 0.
            row['gradient_wait_s'] = row['gradient_received_s'] - row.get('sd_receive_complete_s', row['enqueue_s'])
            check_abort()
            progress()
            row['client_backward_start_s'] = now()
            start = event()
            result = node.backward_job(row['job_id'], processed['cut_gradient'])
            end = event()
            end.synchronize()
            with condition:
                check_abort()
                row['completion_queue_depth_after_backward'] = completions.consumed(row['job_id'])
            row['client_backward_version'] = result['version_before']
            row['cuda_time_s'].update(client_backward=result['backward_cuda_s'], client_optimizer=0.)
            row['_cuda_spans']['client_backward'] = start, end
            row['pending_graph_count_after_backward'] = node.pending_graph_count
            if last:
                deferred_padding[0] = record
            else:
                row['client_phase_padding']['backward'] = pace_phase(
                    wall_origin + row['client_backward_start_s'], record['targets']['backward'], aborted)
                row['client_compute_done_s'] = now()
                row['client_backward_update_s'] = row['client_compute_done_s'] - row['client_backward_start_s']

        def step(records):
            check_abort()
            owner_record = deferred_padding[0]
            row = owner_record['row']
            started = now()
            start = event()
            result = node.step_group()
            end = event()
            end.synchronize()
            ended = now()
            row['client_phase_padding']['backward'] = pace_phase(
                wall_origin + row['client_backward_start_s'], owner_record['targets']['backward'], aborted)
            row['client_compute_done_s'] = now()
            row['client_backward_update_s'] = row['client_compute_done_s'] - row['client_backward_start_s']
            row['client_optimizer_step_performed'] = True
            row['cuda_time_s']['client_optimizer'] = result['optimizer_cuda_s']
            row['_cuda_spans']['client_optimizer'] = start, end
            row['client_idle_wait_s'] = group_idle[0]
            row['client_blocked_on_gradient_s'] = group_idle[0]
            complete = now()
            with condition:
                check_abort()
                for record in records:
                    own = record['row']
                    own.update(group_optimizer_step_start_s=started, group_optimizer_step_end_s=ended,
                        group_complete_s=complete, client_complete_s=complete,
                        client_version_after=result['version_after'])
                    registry.pop(tuple(own['job_id']))
                    pipeline_pending_rows.pop(tuple(own['job_id']))
                    completed_rows.append(own)
                    counters['completed_jobs'] += 1
                last_progress[0] = time.perf_counter()
                condition.notify_all()
            group_idle[0] = 0.
            deferred_padding[0] = None

        _run_pipeline_groups(node, owner._client_batches(images, labels, partitions[cid], round_id, cid),
                             round_id, cid, forward, backward, step, check_abort, wait_ready)

    def server_work(stream):
        while True:
            with condition:
                while (not queue and not aborted.is_set()
                       and (clients_done[0] < count or counters['serviced_jobs'] < expected_jobs)):
                    condition.wait(.05)
                check_abort()
                if not queue:
                    return
                packets = take_ready_batch(queue, args.get('server_max_batch_samples', args['batch_size']),
                                           args.get('server_policy', 'individual'))
                condition.notify_all()
                dequeued = now()
                for index, packet in enumerate(packets):
                    row = packet['row']
                    row['dequeue_s'] = dequeued
                    row['service_seq'] = counters['serviced_jobs'] + index
                    row['queue_wait_s'] = dequeued - row['enqueue_s']
                last_progress[0] = time.perf_counter()
            check_abort()
            progress()
            inputs = []
            for packet in packets:
                if packet['ready'] is not None:
                    stream.wait_event(packet['ready'])
                for key in ('activation', 'labels', 'counts', 'selected_sd', 'selected_labels', 'selected_indices'):
                    if key not in packet:
                        continue
                    packet[key] = packet[key].to(device, non_blocking=True)
                    packet[key].record_stream(stream)
                inputs.append({'job_id': packet['row']['job_id'], 'activation': packet['activation'],
                    'labels': packet['labels'], 'return_cut_gradient': packet['return_cut_gradient'],
                    **{key: packet[key] for key in ('job_kind', 'counts', 'actual_batch_size', 'selected_sd',
                        'selected_labels', 'selected_indices', 'sample_ratio', 'sample_seed') if key in packet}})
            started = now()
            if pipeline:
                with condition:
                    for packet in packets:
                        packet['row']['queue_depth_at_start'] = len(queue) + len(packets)
            server_start = event()
            batch = owner.server.process_batch(inputs)
            gradient_ready = event()
            ended = now()
            batch_row = {'round': round_id, 'batch_id': batch['version_before'],
                'server_version_before': batch['version_before'], 'server_version_after': batch['version_after'],
                'job_ids': [p['row']['job_id'] for p in packets], 'samples': batch['samples'],
                'sd_count': len(packets), 'server_start_s': started, 'server_end_s': ended,
                'forward_cuda_s': batch['forward_cuda_s'], 'backward_cuda_s': batch['backward_cuda_s'],
                'optimizer_cuda_s': batch['optimizer_cuda_s']}
            server_batches.append(batch_row)
            for packet, processed in zip(packets, batch['jobs']):
                row = packet['row']
                row.update(server_start_s=started, server_end_s=ended,
                    server_thread_ident=threading.get_ident(), server_stream_id=int(stream.cuda_stream),
                    server_batch_id=batch['version_before'], server_batch_samples=batch['samples'],
                    server_batch_sd_count=len(packets), loss=processed['loss'], correct=processed['correct'],
                    server_version_before=batch['version_before'], server_version_after=batch['version_after'],
                    gradient_scale_to_local_mean=processed['gradient_scale_to_local_mean'],
                    gradient_bytes=_float_bytes(processed['cut_gradient']))
                row['cuda_time_s'].update({name: batch[key] if len(packets) == 1 else None
                    for name, key in (('server_forward', 'forward_cuda_s'), ('server_backward', 'backward_cuda_s'),
                                      ('server_optimizer', 'optimizer_cuda_s'))})
                if 'job_kind' in packet:
                    row.update({key: value for key, value in processed.items() if key not in (
                        'cut_gradient', 'job_id', 'samples', 'version_before', 'version_after', 'prototype_compute_s')})
                    row['server_prototype_compute_s'] = processed['prototype_compute_s']
                    row['server_model_compute_s'] = sum(batch[key] for key in (
                        'forward_cuda_s', 'backward_cuda_s', 'optimizer_cuda_s'))
                if len(packets) == 1:
                    row['_cuda_spans']['server'] = server_start, gradient_ready
                if packet['future'] is not None:
                    if transport is None:
                        with condition:
                            check_abort()
                            if pipeline:
                                row['gradient_available_s'] = now()
                                row['completion_queue_depth_on_available'] = completions.available(row['job_id'])
                                row['max_completion_queue_depth'] = completions.peaks[row['client_id']]
                            packet['future'].set_result((processed, gradient_ready))
                    else:
                        gradient = _stage_cpu(processed['cut_gradient'])
                        staged = event()
                        staged.synchronize()
                        transport.send_gradient(packet, gradient)
            with condition:
                counters['serviced_jobs'] += len(packets)
                counters['serviced_batches'] += 1
                last_progress[0] = time.perf_counter()
                check_abort()
                condition.notify_all()
            if counters['serviced_batches'] % 100 == 0 and hasattr(owner, '_log'):
                owner._log('live_async server_step=' + str(batch['version_after']))
            del packets, packet, processed, row, inputs, batch

    def worker(stream, cid=None):
        try:
            with torch.cuda.device(device), torch.cuda.stream(stream), torch.enable_grad():
                stream.wait_event(main_ready)
                start_barrier.wait(timeout=timeout)
                if cid is None:
                    server_work(stream)
                else:
                    (pipeline_client_work if pipeline else client_work)(cid, stream)
        except _Aborted:
            pass
        except BaseException as error:
            cancel(error)
        finally:
            try:
                done = torch.cuda.Event()
                done.record(stream)
                with condition:
                    stream_events[id(stream)] = done
            except BaseException as error:
                cancel(error)
            with condition:
                if cid is not None:
                    clients_done[0] += 1
                last_progress[0] = time.perf_counter()
                condition.notify_all()

    threads.append(threading.Thread(target=worker, args=(server_stream,),
        name=f"na-sfl-server-{round_id}", daemon=True))
    threads.extend(threading.Thread(target=worker, args=(stream, cid),
        name=f"na-sfl-client-{round_id}-{cid}", daemon=True) for cid, stream in enumerate(client_streams))
    cleanup_deadline = None
    try:
        if getattr(owner, 'network', None) is not None:
            from utils.na_sfl_transport import TCPRoundTransport
            transport = TCPRoundTransport(owner.network.create_connections(), admit, receive_gradient,
                cancel, now, capacity=2 if pipeline else args['network']['send_buffer_sd'], worker_timeout_s=timeout).start()
        for thread in threads:
            thread.start()
            started_threads.append(thread)
        start_barrier.wait(timeout=timeout)
        while any(thread.is_alive() for thread in started_threads):
            if not aborted.is_set() and time.perf_counter() - last_progress[0] > timeout:
                cancel(TimeoutError("live_async worker made no cooperative progress"))
            if aborted.is_set() and cleanup_deadline is None:
                cleanup_deadline = time.perf_counter() + timeout
            if cleanup_deadline is not None and time.perf_counter() >= cleanup_deadline:
                break
            for thread in started_threads:
                thread.join(timeout=.02)
    except BaseException as error:
        cancel(error)
    finally:
        if any(thread.is_alive() for thread in started_threads):
            deadline = cleanup_deadline or time.perf_counter() + timeout
            for thread in started_threads:
                thread.join(timeout=max(0., deadline - time.perf_counter()))
        workers_alive = sum(thread.is_alive() for thread in started_threads)
        if transport is not None:
            try:
                if aborted.is_set():
                    transport.abort()
                transport.finish()
            except BaseException as error:
                cancel(error)
        transport_stats = transport.stats() if transport is not None else None
        streams_completed = False
        if not workers_alive:
            try:
                deadline = time.perf_counter() + timeout
                events = list(stream_events.values())
                while not all(value.query() for value in events):
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("live_async CUDA stream cleanup timed out")
                    time.sleep(.01)
                streams_completed = len(events) == len(started_threads)
                for value in events:
                    main_stream.wait_event(value)
            except BaseException as error:
                cancel(error)
        if aborted.is_set() and streams_completed:
            for client in owner.clients:
                if hasattr(client, 'clear_pending'):
                    client.clear_pending()
                else:
                    client.pending = None
        with condition:
            if aborted.is_set():
                queue.clear()
            unresolved = sum(not future.done() for future in registry.values())
        for row in completed_rows:
            for phase, (start, end) in row.pop("_cuda_spans").items():
                if streams_completed:
                    row[f"cuda_{phase}_start_s"] = float(origin.elapsed_time(start) / 1000.)
                    row[f"cuda_{phase}_end_s"] = float(origin.elapsed_time(end) / 1000.)
        rows = sorted(completed_rows, key=lambda row: row.get('arrival_seq', math.inf))
        pending_graphs = sum(client.pending is not None for client in owner.clients)
        success = (not failures and not workers_alive and streams_completed and not queue
                   and not unresolved and not pending_graphs
                   and counters["admitted_jobs"] == counters["serviced_jobs"] == counters["completed_jobs"])
        if pipeline:
            success = success and not completions.pending and not pipeline_pending_rows
        if transport_stats is not None:
            success = success and all(transport_stats[key] == 0 for key in
                                     ('workers_alive', 'pending_sd', 'pending_gradients'))
        owner.last_async_batches = server_batches
        owner.last_async_boundary = {
            "round": round_id, "success": success, "queue_empty": not queue,
            "wall_origin_perf_counter_s":wall_origin,
            "unresolved_futures": unresolved, "workers_alive": workers_alive,
            "pending_graphs": pending_graphs, "streams_completed": streams_completed,
            "trace_sha256": async_trace_sha256(rows), "runtimecounts": dict(counters),
            'server_batch_trace_sha256': async_trace_sha256(server_batches),
            "client_workers": count, "server_workers": 1,
            "client_streams": count, "server_streams": 1,
            'transport': transport_stats,
        }
        if pipeline:
            owner.last_async_boundary['pipeline'] = {
                'max_completion_queue_depth_per_client': list(completions.peaks),
                'completion_queue_depth_per_client': list(completions.depth),
                'pending_completed_gradients': len(completions.pending),
                'pending_job_rows': len(pipeline_pending_rows)}
    if not owner.last_async_boundary["success"]:
        error = failures[0] if failures else RuntimeError("incomplete live_async boundary")
        raise RuntimeError(f"live_async round failed: {type(error).__name__}: {error}") from error
    for row in rows:
        owner.job_records.append(row)
    return rows

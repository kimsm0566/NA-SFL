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

import torch
from utils.na_sfl_batch import take_ready_batch, pace_phase, phase_targets

def async_trace_sha256(rows):
    """Hash the complete, JSON-serializable server-ordered round trace."""
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class _Aborted(Exception):
    pass


def run_async_round(owner, images, labels, partitions, round_id):
    """Run NA-SFL Client/Server workers and commit after full CUDA cleanup.

    Admission follows actual activation readiness. worker_timeout_s detects a
    lack of cooperative progress; it does not limit the total round duration.
    """
    count = len(owner.clients)
    args = owner.args
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
    origin.record(main_stream)
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
            if hasattr(node, 'prepare_upload'):
                upload = node.prepare_upload(job_id, begun, batch_labels)
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
                packet.update({key: upload[key] for key in ('activation', 'labels', 'job_kind', 'counts') if key in upload})
                row.update(run_id=owner.run_dir.name, job_kind=upload['job_kind'],
                    prototype_compute_s=upload['prototype_compute_s'],
                    client_forward_compute_s=begun['forward_cuda_s'],
                    activation_bytes=packet['activation'].numel() * packet['activation'].element_size())
                packet['actual_batch_size'] = len(sample_ids)
            with condition:
                if future is not None:
                    registry[job_id] = future
            if transport is None:
                admit(packet)
            else:
                row['sd_host_stage_start_s'] = now()
                host_activation = torch.empty_like(packet['activation'], device='cpu', pin_memory=True)
                host_labels = torch.empty_like(packet['labels'], device='cpu', pin_memory=True)
                host_activation.copy_(packet['activation'], non_blocking=True)
                host_labels.copy_(packet['labels'], non_blocking=True)
                if 'counts' in packet:
                    host_counts = torch.empty_like(packet['counts'], device='cpu', pin_memory=True)
                    host_counts.copy_(packet['counts'], non_blocking=True)
                    packet['counts'] = host_counts
                staged = event()
                staged.synchronize()
                row['sd_host_stage_end_s'] = now()
                packet.update(activation=host_activation, labels=host_labels, ready=None)
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
                processed['cut_gradient'] = processed['cut_gradient'].to(device, non_blocking=True)
            processed["cut_gradient"].record_stream(stream)
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
                for key in ('activation', 'labels', *(['counts'] if 'counts' in packet else [])):
                    packet[key] = packet[key].to(device, non_blocking=True)
                    packet[key].record_stream(stream)
                inputs.append({'job_id': packet['row']['job_id'], 'activation': packet['activation'],
                    'labels': packet['labels'], 'return_cut_gradient': packet['return_cut_gradient'],
                    **{key: packet[key] for key in ('job_kind', 'counts', 'actual_batch_size') if key in packet}})
            started = now()
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
                    gradient_bytes=(processed['cut_gradient'].numel() * processed['cut_gradient'].element_size()
                                    if processed['cut_gradient'] is not None else 0))
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
                            packet['future'].set_result((processed, gradient_ready))
                    else:
                        gradient = torch.empty_like(processed['cut_gradient'], device='cpu', pin_memory=True)
                        gradient.copy_(processed['cut_gradient'], non_blocking=True)
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
                    client_work(cid, stream)
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
                cancel, now, capacity=args['network']['send_buffer_sd'], worker_timeout_s=timeout).start()
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
    if not owner.last_async_boundary["success"]:
        error = failures[0] if failures else RuntimeError("incomplete live_async boundary")
        raise RuntimeError(f"live_async round failed: {type(error).__name__}: {error}") from error
    for row in rows:
        owner.job_records.append(row)
    return rows

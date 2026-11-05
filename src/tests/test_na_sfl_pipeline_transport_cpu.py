"""C transport instrumentation and real full-duplex socket contracts."""
import socket
import threading
import time

import pytest
import torch

from utils.na_sfl_transport import TCPRoundTransport


def item(index):
    return {'row': {'job_id': (1, 0, index), 'client_id': 0, 'pipeline_group_id': 0},
            'activation': torch.ones(2, 3) * index, 'labels': torch.tensor([0, 1]),
            'return_cut_gradient': True, 'future': object(), 'ready': None}


def until(predicate):
    end = time.monotonic() + 2
    while not predicate() and time.monotonic() < end:
        time.sleep(.002)
    assert predicate()


def test_two_tcp_jobs_fifo_upload_and_identity_reordered_download_with_bounded_metrics():
    accepted, received, errors = [], [], []
    origin = time.monotonic()
    transport = TCPRoundTransport([socket.socketpair()], accepted.append,
        lambda job, gradient: received.append((job, gradient)), errors.append,
        lambda: time.monotonic() - origin, capacity=2, worker_timeout_s=2).start()
    packets = [item(i) for i in range(2)]
    try:
        for packet in packets:
            transport.reserve(0, lambda: None)
            transport.submit(packet)
        until(lambda: len(accepted) == 2)
        assert [p['row']['job_id'] for p in accepted] == [(1, 0, 0), (1, 0, 1)]
        for packet in reversed(accepted):
            transport.send_gradient(packet, packet['activation'] + 10)
        transport.finish()
        assert [job for job, _ in received] == [(1, 0, 1), (1, 0, 0)]
        stats = transport.stats()
        assert stats['pipeline']['max_outstanding_jobs_per_client'] == [2]
        assert stats['pipeline']['max_sender_queue_depth_per_client'][0] <= 2
        assert stats['pipeline']['max_gradient_receive_queue_depth_per_client'][0] <= 2
        assert stats['pipeline']['sender_workers_per_client'] == 1
        assert stats['pipeline']['queued_cpu_bytes'] == 0
        for packet in packets:
            row = packet['row']
            assert row['sendqueue_enqueue_s'] <= row['sender_dequeue_s'] <= row['sd_send_start_s']
            assert row['gradient_receive_start_s'] <= row['gradient_receive_complete_s']
            assert row['gradient_available_s'] >= row['gradient_receive_complete_s']
        assert errors == []
    finally:
        transport.abort()


def test_pipeline_third_job_rejected_before_snapshot_even_after_sd_credit_releases():
    accepted, errors = [], []
    transport = TCPRoundTransport([socket.socketpair()], accepted.append,
        lambda *_: None, errors.append, time.monotonic,
        capacity=2, worker_timeout_s=2).start()
    try:
        for index in range(2):
            transport.reserve(0, lambda: None)
            transport.submit(item(index))
        until(lambda: len(accepted) == 2)
        transport.reserve(0, lambda: None)
        with pytest.raises(RuntimeError, match='pipeline.*2|pipeline.*full'):
            transport.submit(item(2))
    finally:
        transport.abort()
        with pytest.raises(RuntimeError):
            transport.finish()
    assert transport.stats()['workers_alive'] == 0
    assert transport.stats()['pipeline']['queued_cpu_bytes'] == 0


def test_second_upload_blocked_while_first_download_completes_on_same_duplex_socket():
    client, server = socket.socketpair()
    class GateSecondFrame:
        def __init__(self):
            self.calls = 0
            self.entered, self.release = threading.Event(), threading.Event()
        def __getattr__(self, name): return getattr(client, name)
        def sendall(self, raw):
            self.calls += 1
            # length, JSON, activation, labels = four sends for first frame.
            if self.calls == 5:
                self.entered.set()
                if not self.release.wait(2): raise TimeoutError('UL gate')
            client.sendall(raw)
        def shutdown(self, how):
            self.release.set()
            return client.shutdown(how)
    gate = GateSecondFrame()
    accepted, completed, errors = [], [], []
    transport = TCPRoundTransport([(gate, server)], accepted.append,
        lambda job, gradient: completed.append(job), errors.append,
        time.monotonic, capacity=2, worker_timeout_s=2).start()
    try:
        for index in range(2):
            transport.reserve(0, lambda: None)
            transport.submit(item(index))
        assert gate.entered.wait(1)
        until(lambda: len(accepted) == 1)
        transport.send_gradient(accepted[0], torch.ones(2, 3))
        until(lambda: completed == [(1, 0, 0)])
        assert not gate.release.is_set() and len(accepted) == 1
        gate.release.set()
        until(lambda: len(accepted) == 2)
        transport.send_gradient(accepted[1], torch.ones(2, 3))
        transport.finish()
        assert completed == [(1, 0, 0), (1, 0, 1)] and errors == []
    finally:
        transport.abort()


def test_pipeline_partial_socket_reads_and_duplicate_gradient_abort_once():
    client, server = socket.socketpair()
    class Fragments:
        def __getattr__(self, name): return getattr(server, name)
        def recv_into(self, view): return server.recv_into(view[:3])
    accepted, completed, errors = [], [], []
    transport = TCPRoundTransport([(client, Fragments())], accepted.append,
        lambda job, gradient: completed.append(job), errors.append,
        time.monotonic, capacity=2, worker_timeout_s=2).start()
    try:
        transport.reserve(0, lambda: None)
        packet = item(0)
        transport.submit(packet)
        until(lambda: len(accepted) == 1)
        transport.send_gradient(packet, torch.ones(2, 3))
        with pytest.raises(ValueError, match='duplicate|no cut gradient'):
            transport.send_gradient(packet, torch.ones(2, 3))
        transport.finish()
        assert completed == [(1, 0, 0)] and errors == []
    finally:
        transport.abort()
    assert transport.stats()['workers_alive'] == 0


def test_final_completion_can_start_next_group_without_stale_record_window_race():
    accepted, completed, errors = [], [], []
    def complete(job, gradient):
        completed.append(job)
        if job == (1, 0, 1):
            # The Trainer may wake and enqueue its next complete group before
            # the Receiver callback returns. Old completed payloads free first.
            for index in (2, 3):
                transport.reserve(0, lambda: None)
                transport.submit(item(index))
    transport = TCPRoundTransport([socket.socketpair()], accepted.append,
        complete, errors.append, time.monotonic, capacity=2, worker_timeout_s=2).start()
    try:
        for index in (0, 1):
            transport.reserve(0, lambda: None)
            transport.submit(item(index))
        until(lambda: len(accepted) == 2)
        for packet in list(accepted):
            transport.send_gradient(packet, torch.ones(2, 3))
        until(lambda: len(accepted) == 4 or bool(errors))
        assert errors == []
        for packet in accepted[2:]:
            transport.send_gradient(packet, torch.ones(2, 3))
        transport.finish()
        assert completed == [(1, 0, i) for i in range(4)]
        assert transport.stats()['pipeline']['max_outstanding_jobs_per_client'] == [2]
    finally:
        transport.abort()


def test_completed_future_queue_measures_unconsumed_gradients_after_receiver_dequeues():
    from concurrent.futures import Future
    from utils.deit_async import _PipelineCompletions
    completions = _PipelineCompletions(1)
    futures = {(1, 0, i): Future() for i in range(2)}
    accepted, errors = [], []
    def receive(job, gradient):
        completions.available(job)
        futures[job].set_result(gradient)
    transport = TCPRoundTransport([socket.socketpair()], accepted.append,
        receive, errors.append, time.monotonic, capacity=2, worker_timeout_s=2).start()
    try:
        for index in range(2):
            transport.reserve(0, lambda: None)
            transport.submit(item(index))
        until(lambda: len(accepted) == 2)
        # Queue only one DL at a time. Both Futures remain unconsumed after
        # the Receiver dequeued and delivered each of the two responses.
        for packet in accepted:
            job = packet['row']['job_id']
            transport.send_gradient(packet, torch.ones(2, 3))
            until(lambda: futures[job].done())
        transport.finish()
        assert completions.depth == completions.peaks == [2]
        assert transport.stats()['pipeline']['max_gradient_receive_queue_depth_per_client'] == [1]
        assert 'max_completion_queue_depth_per_client' not in transport.stats()['pipeline']
        completions.consumed((1, 0, 1))
        assert completions.depth == [1]
        completions.consumed((1, 0, 0))
        assert completions.depth == [0] and completions.peaks == [2]
        assert errors == []
    finally:
        transport.abort()


def test_completed_gradient_queue_rejects_third_duplicate_and_abort_clears_pending():
    from utils.deit_async import _PipelineCompletions
    completions = _PipelineCompletions(1)
    completions.available((1, 0, 0))
    completions.available((1, 0, 1))
    with pytest.raises(RuntimeError, match='duplicate'):
        completions.available((1, 0, 1))
    with pytest.raises(RuntimeError, match='full'):
        completions.available((1, 0, 2))
    completions.clear()
    assert completions.depth == [0] and completions.peaks == [2]
    with pytest.raises(RuntimeError, match='unavailable'):
        completions.consumed((1, 0, 0))

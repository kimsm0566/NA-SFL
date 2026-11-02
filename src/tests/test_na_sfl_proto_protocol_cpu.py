"""Typed Prototype wire contracts; real TCP framing without model computation."""
import socket
import threading
import time

import pytest
import torch

from utils import na_sfl_transport as transport


def packet(kind='PROTO_JOB', batch=0):
    return {'job_kind': kind, 'row': {'job_id': (1, 0, batch), 'client_id': 0,
            'run_id': 'typed-test', 'samples': 3},
            'activation': torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2),
            'labels': torch.tensor([1, 3]), 'counts': torch.tensor([2, 1]),
            'return_cut_gradient': True}


def test_proto_request_contains_no_original_sd_or_sample_labels():
    from utils.na_sfl_proto_protocol import request_message, validate_message
    tensors, meta = request_message(packet())
    assert set(tensors) == {'prototypes', 'classes', 'counts'}
    assert meta['kind'] == 'PROTO_JOB' and meta['actual_batch_size'] == 3
    assert meta['job_id'] == 'typed-test:1:0:0'
    validate_message(tensors, meta)
    assert sum(x.numel() * x.element_size() for x in tensors.values()) == 96


@pytest.mark.parametrize('bad', ['counts_sum', 'counts_zero', 'classes_order', 'nonfinite', 'dtype', 'identity'])
def test_typed_request_rejects_invalid_proto_or_identity(bad):
    from utils.na_sfl_proto_protocol import request_message
    p = packet()
    if bad == 'counts_sum': p['counts'] = torch.tensor([1, 1])
    if bad == 'counts_zero': p['counts'] = torch.tensor([3, 0])
    if bad == 'classes_order': p['labels'] = torch.tensor([3, 1])
    if bad == 'nonfinite': p['activation'][0, 0, 0, 0] = float('nan')
    if bad == 'dtype': p['activation'] = p['activation'].double()
    if bad == 'identity': p['row']['job_id'] = (1, 1, 0)
    with pytest.raises(ValueError): request_message(p)


def test_typed_proto_and_sd_roundtrip_use_received_tensors_and_matching_replies():
    pairs = [socket.socketpair()]
    for sock in pairs[0]: sock.settimeout(3)
    received, gradients, errors = [], [], []
    done = threading.Event()
    channel = None
    def admitted(p):
        received.append(p)
        channel.send_gradient(p, torch.full_like(p['activation'], .25))
    def returned(job, gradient):
        gradients.append((job, gradient))
        done.set()
    channel = transport.TCPRoundTransport(pairs, admitted, returned, errors.append,
        time.perf_counter, capacity=1, worker_timeout_s=3).start()
    try:
        for kind, index in (('PROTO_JOB', 0), ('SD_JOB', 1)):
            p = packet(kind, index)
            if kind == 'SD_JOB':
                p['row']['samples'] = 2
                p.pop('counts')
            done.clear()
            channel.reserve(0, lambda: False)
            channel.submit(p)
            assert done.wait(3), errors
            assert not errors
            decoded = received[-1]
            assert decoded['activation'].data_ptr() != p['activation'].data_ptr()
            assert torch.equal(decoded['activation'], p['activation'])
            assert decoded['job_kind'] == kind
            assert torch.equal(gradients[-1][1], torch.full_like(p['activation'], .25))
            assert p['row']['request_kind'] == kind
            assert p['row']['response_kind'] == ('PROTO_GRADIENT' if kind == 'PROTO_JOB' else 'SD_GRADIENT')
        channel.finish()
        stats = channel.stats()
        assert stats['frames_sent'] == stats['frames_received'] == 4
        assert stats['pending_sd'] == stats['pending_gradients'] == stats['workers_alive'] == 0
    finally:
        if not channel._finished:
            channel.abort()
            try:
                channel.finish()
            except RuntimeError:
                pass


def test_receive_timestamp_precedes_validation_and_fcfs_survives_validation_delay(monkeypatch):
    from utils import na_sfl_proto_protocol as protocol
    pairs = [socket.socketpair(), socket.socketpair()]
    for pair in pairs:
        for sock in pair: sock.settimeout(3)
    entered, release, second_validated, done = (threading.Event() for _ in range(4))
    admitted, replies, errors = [], [], []
    original = protocol.validate_message
    def delayed(tensors, metadata):
        original(tensors, metadata)
        if metadata['kind'] == 'PROTO_JOB':
            if metadata['client_id'] == 0:
                entered.set()
                assert release.wait(3)
            else:
                second_validated.set()
    # Requests are built before patching validation to isolate receive validation.
    packets = [packet(), packet()]
    packets[1]['row']['client_id'] = 1
    packets[1]['row']['job_id'] = (1, 1, 0)
    channel = None
    def admit(p):
        admitted.append(p['row']['client_id'])
        channel.send_gradient(p, torch.zeros_like(p['activation']))
    def returned(job_id, gradient):
        replies.append(job_id)
        if len(replies) == 2: done.set()
    channel = transport.TCPRoundTransport(pairs, admit, returned, errors.append,
        time.perf_counter, capacity=1, worker_timeout_s=3).start()
    try:
        # Submit calls request_message/validation too; delay only receive threads.
        def receiving_only(tensors, metadata):
            if threading.current_thread().name.endswith('sd_receive'):
                delayed(tensors, metadata)
            else:
                original(tensors, metadata)
        monkeypatch.setattr(protocol, 'validate_message', receiving_only)
        channel.reserve(0, lambda: False); channel.submit(packets[0])
        assert entered.wait(3), errors
        assert 'sd_receive_complete_s' in packets[0]['row']
        channel.reserve(1, lambda: False); channel.submit(packets[1])
        assert second_validated.wait(3), errors
        assert admitted == []
        release.set()
        assert done.wait(3), errors
        assert admitted == [0, 1] and not errors
        channel.finish()
    finally:
        release.set()
        if not channel._finished:
            channel.abort()
            try: channel.finish()
            except RuntimeError: pass

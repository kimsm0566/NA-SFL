"""Hybrid typed frames and actual CPU socket transfers, without model work."""
import socket
import threading
import time

import pytest
import torch

from utils.na_sfl_proto_protocol import request_message, reply_message, validate_message
from utils.na_sfl_transport import TCPRoundTransport, send_tensors, recv_tensors


def packet():
    return {'job_kind': 'HYBRID_JOB', 'row': {'job_id': (1, 0, 0), 'client_id': 0,
        'run_id': 'hybrid-test', 'samples': 8}, 'return_cut_gradient': True,
        'activation': torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2),
        'labels': torch.tensor([1, 3]), 'counts': torch.tensor([4, 4]),
        'selected_sd': torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2) + 20,
        'selected_labels': torch.tensor([1, 3]), 'selected_indices': torch.tensor([2, 6], dtype=torch.int32),
        'actual_batch_size': 8, 'sample_ratio': .25, 'sample_seed': 42}


def gradient(p):
    return {'grad_selected_sd': torch.full_like(p['selected_sd'], .8),
        'grad_prototypes': torch.full_like(p['activation'], .2),
        'selected_indices': p['selected_indices'].clone(), 'classes': p['labels'].clone()}


def test_hybrid_request_and_reply_are_single_typed_frames():
    p = packet()
    tensors, metadata = request_message(p)
    assert set(tensors) == {'prototypes', 'classes', 'counts', 'selected_sd', 'selected_labels', 'selected_indices'}
    assert metadata['sample_ratio'] == .25 and metadata['actual_batch_size'] == 8
    reply, meta = reply_message(metadata, gradient(p))
    assert meta['kind'] == 'HYBRID_GRADIENT' and set(reply) == set(gradient(p))
    validate_message(reply, meta)


@pytest.mark.parametrize('bad', ['duplicate', 'range', 'index_dtype', 'wrong_m', 'label', 'label_count',
    'nan_sd', 'nan_proto', 'count_sum', 'float_ratio', 'missing', 'classes_order'])
def test_request_rejects_invalid_hybrid_tensor_contract(bad):
    p = packet()
    if bad == 'duplicate': p['selected_indices'][:] = 2
    if bad == 'range': p['selected_indices'][1] = 8
    if bad == 'index_dtype': p['selected_indices'] = p['selected_indices'].long()
    if bad == 'wrong_m': p['sample_ratio'] = .5
    if bad == 'label': p['selected_labels'][0] = 2
    if bad == 'label_count': p['counts'] = torch.tensor([1, 7]); p['selected_labels'][:] = 1
    if bad == 'nan_sd': p['selected_sd'][0] = float('nan')
    if bad == 'nan_proto': p['activation'][0] = float('nan')
    if bad == 'count_sum': p['counts'] = torch.tensor([2, 3])
    if bad == 'float_ratio': p['sample_ratio'] = float('nan')
    if bad == 'missing': p.pop('selected_labels')
    if bad == 'classes_order': p['labels'] = torch.tensor([3, 1])
    with pytest.raises(ValueError): request_message(p)


def test_actual_socket_roundtrip_snapshots_and_decodes_every_tensor(monkeypatch):
    from utils import na_sfl_transport as transport
    p = packet()
    expected = {key: value.clone() for key, value in p.items() if isinstance(value, torch.Tensor)}
    admitted, returned, errors = [], [], []
    release, entered, done = (threading.Event() for _ in range(3))
    original_send = transport.send_tensors
    def held_send(sock, tensors, metadata):
        if metadata['kind'] == 'HYBRID_JOB':
            entered.set()
            assert release.wait(3)
        return original_send(sock, tensors, metadata)
    monkeypatch.setattr(transport, 'send_tensors', held_send)
    channel = None
    def on_sd(decoded):
        admitted.append(decoded)
        for key, expected_tensor in expected.items():
            assert torch.equal(decoded[key], expected_tensor)
            assert decoded[key].data_ptr() != p[key].data_ptr()
        channel.send_gradient(decoded, gradient(decoded))
    def on_gradient(job, g):
        returned.append((job, g)); done.set()
    channel = TCPRoundTransport([socket.socketpair()], on_sd, on_gradient, errors.append,
        time.perf_counter, capacity=1, worker_timeout_s=3).start()
    try:
        channel.reserve(0, lambda: False); channel.submit(p)
        assert entered.wait(3)
        for key in expected: p[key].zero_()
        release.set()
        assert done.wait(3), errors
        assert not errors and len(admitted) == len(returned) == 1
        torch.testing.assert_close(returned[0][1]['grad_selected_sd'], torch.full_like(expected['selected_sd'], .8))
        assert torch.equal(returned[0][1]['selected_indices'], expected['selected_indices'])
        channel.finish()
        stats = channel.stats()
        assert stats['frames_sent'] == stats['frames_received'] == 2
        assert stats['sd_payload_bytes_sent'] == stats['sd_payload_bytes_received'] == 184
        assert stats['gradient_payload_bytes_sent'] == stats['gradient_payload_bytes_received'] == 152
        assert stats['pending_sd'] == stats['pending_gradients'] == stats['workers_alive'] == 0
    finally:
        release.set()
        if not channel._finished:
            channel.abort()
            try: channel.finish()
            except RuntimeError: pass


def test_reply_rejects_wrong_echo_before_any_downlink_enqueue():
    p = packet()
    errors = []
    done = threading.Event()
    def on_sd(decoded):
        g = gradient(decoded); g['selected_indices'] = g['selected_indices'].flip(0)
        with pytest.raises(ValueError): channel.send_gradient(decoded, g)
        channel.send_gradient(decoded, gradient(decoded))
    channel = TCPRoundTransport([socket.socketpair()], on_sd, lambda *args: done.set(), errors.append,
        time.perf_counter, capacity=1, worker_timeout_s=3).start()
    try:
        channel.reserve(0, lambda: False); channel.submit(p)
        assert done.wait(3), errors
        assert not errors
        channel.finish()
    finally:
        if not channel._finished:
            channel.abort()
            try: channel.finish()
            except RuntimeError: pass


def test_hybrid_frames_support_fragmented_receipt_and_timeout():
    p = packet()
    tensors, meta = request_message(p)
    a, b = socket.socketpair()
    class Fragments:
        def recv_into(self, view): return b.recv_into(view[:3])
    try:
        a.settimeout(2); b.settimeout(2)
        sender = threading.Thread(target=send_tensors, args=(a, tensors, meta))
        sender.start()
        received, metadata = recv_tensors(Fragments())
        sender.join(2)
        validate_message(received, metadata)
        assert torch.equal(received['selected_sd'], p['selected_sd'])
        b.settimeout(.01)
        with pytest.raises(socket.timeout): recv_tensors(b)
    finally:
        a.close(); b.close()


@pytest.mark.parametrize('tcp', [False, True])
@pytest.mark.parametrize('fail_server', [False, True])
def test_real_worker_orchestration_two_branches_single_inflight_and_cleanup(tmp_path, monkeypatch, tcp, fail_server):
    """Exercise actual workers; replace only CUDA hardware APIs with CPU stubs."""
    import torch._dynamo
    from contextlib import nullcontext
    from types import SimpleNamespace
    from torch import nn
    from utils import sflv2, na_sfl_prototype, na_sfl_proto_client, na_sfl_hybrid
    from utils.deit_async import run_async_round
    from utils.na_sfl_config import load_config
    class Event:
        def __init__(self, **kwargs): pass
        def record(self, *args): pass
        def synchronize(self): pass
        def query(self): return True
        def elapsed_time(self, other): return 0.
    class Stream:
        cuda_stream = 1
        def __init__(self, **kwargs): pass
        def wait_event(self, event): pass
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'Stream', Stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: Stream())
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    original_to, original_empty = torch.Tensor.to, torch.empty_like
    def cpu_to(tensor, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)) and str(args[0]).startswith('cuda'):
            return tensor
        return original_to(tensor, *args, **kwargs)
    def cpu_empty(tensor, *args, **kwargs):
        kwargs.pop('pin_memory', None)
        return original_empty(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    monkeypatch.setattr(torch, 'empty_like', cpu_empty)
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda *args: None)
    monkeypatch.setattr(sflv2, '_model_device', lambda model: torch.device('cpu'))
    for module in (sflv2, na_sfl_prototype, na_sfl_proto_client, na_sfl_hybrid):
        monkeypatch.setattr(module, '_timed', lambda device, fn: (fn(), 0.))
    cfg = load_config(overrides={'method': 'na_sfl_proto_random_sd', 'n_clients': 2,
        'slow_client_ids': [0], 'batch_size': 8, 'samples_per_client': 16,
        'random_sd': {'sample_ratio': .25, 'sample_seed': 42}, 'worker_timeout_s': 3.})
    if tcp:
        cfg['network'] = {'send_buffer_sd': 3}
    server = na_sfl_prototype.PrototypeServer(nn.Sequential(nn.Conv2d(2, 10, 1),
        nn.AdaptiveAvgPool2d(1), nn.Flatten()), {**cfg, 'boundary_shape_per_sample': (2, 2, 2)})
    nodes = [na_sfl_hybrid.RandomSDClient(i, nn.Conv2d(1, 2, 1), cfg, prototype_mode=i == 0) for i in range(2)]
    if fail_server:
        def broken(packets): raise ValueError('intentional server failure')
        server.process_batch = broken
    network = SimpleNamespace(create_connections=lambda: [socket.socketpair() for _ in nodes]) if tcp else None
    owner = SimpleNamespace(args=cfg, device='cuda:0', clients=nodes, server=server,
        isolated_augmentation=True, run_dir=tmp_path, job_records=[], network=network, _log=lambda text: None)
    def batches(images, labels, partition, round_id, cid):
        for batch in range(2):
            # The next yield executes only after the preceding single in-flight job completed.
            assert nodes[cid].pending is None and nodes[cid]._hybrid_pending is None
            yield batch, (torch.ones(8, 1, 2, 2) * (1 + batch), torch.tensor([1]*4+[3]*4),
                          torch.arange(batch*8, batch*8+8))
    owner._client_batches = batches
    server.start_round(1)
    if fail_server:
        with pytest.raises(RuntimeError, match='intentional server failure'):
            run_async_round(owner, None, None, [list(range(16))]*2, 1)
        assert owner.job_records == []
        assert owner.last_async_boundary['pending_graphs'] == 0
        assert all(node.pending is None and node._hybrid_pending is None for node in nodes)
    else:
        rows = run_async_round(owner, None, None, [list(range(16))]*2, 1)
        assert len(rows) == server.version == 4
        assert all(node.version == 2 and node.pending is None and node._hybrid_pending is None for node in nodes)
        hybrid_rows = [row for row in rows if row['job_kind'] == 'HYBRID_JOB']
        assert len(hybrid_rows) == 2 and all(row['selected_sd_count'] == 2 for row in hybrid_rows)
        assert all(row['activation_bytes'] == row['gradient_bytes'] == 4*2*2*2*4 for row in hybrid_rows)
        assert owner.last_async_boundary['success']
        if tcp:
            stats = owner.last_async_boundary['transport']
            assert stats['frames_sent'] == stats['frames_received'] == 8
            assert stats['sd_payload_bytes_sent'] == stats['sd_payload_bytes_received']
            assert stats['gradient_payload_bytes_sent'] == stats['gradient_payload_bytes_received']
            assert stats['pending_sd'] == stats['pending_gradients'] == stats['workers_alive'] == 0
    assert not torch.cuda.is_initialized()

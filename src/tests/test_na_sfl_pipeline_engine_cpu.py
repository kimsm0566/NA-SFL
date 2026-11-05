"""CPU scheduling contracts for the same group loop used by the CUDA actor."""
from concurrent.futures import Future

import pytest


def test_group_loop_submits_second_forward_before_consuming_eager_first_gradient():
    from utils.deit_async import _run_pipeline_groups
    events, versions, futures = [], [], {}

    class Node:
        version = 0
        def begin_group(self, ids):
            events.append(('group', tuple(ids)))

    node = Node()
    def forward(batch, job_id, size):
        events.append(('forward', job_id[2], node.version, size))
        versions.append(node.version)
        future = futures[job_id] = Future()
        # The first completion arrives while the second forward is outstanding.
        if job_id[2] % 2 == 0:
            future.set_result(job_id)
        else:
            future.set_result(job_id)
        return {'job_id': job_id, 'future': future}
    def wait_ready(records):
        return list(reversed(records))
    def backward(record, last):
        events.append(('backward', record['job_id'][2], last))
    def step(records):
        events.append(('step', len(records)))
        node.version += 1
    _run_pipeline_groups(node, enumerate(range(5)), 7, 0, forward, backward, step,
                         lambda: None, wait_ready)
    assert versions == [0, 0, 1, 1, 2]
    assert events[:5] == [('group', ((7, 0, 0), (7, 0, 1))),
                          ('forward', 0, 0, 2), ('forward', 1, 0, 2),
                          ('backward', 1, False), ('backward', 0, True)]
    assert [e for e in events if e[0] == 'step'] == [('step', 2), ('step', 2), ('step', 1)]


def test_group_loop_stops_without_step_when_backward_fails():
    from utils.deit_async import _run_pipeline_groups
    class Node:
        def begin_group(self, ids):
            pass
    def fail(record, last):
        raise ValueError('bad gradient')
    steps = []
    with pytest.raises(ValueError, match='bad gradient'):
        _run_pipeline_groups(Node(), enumerate(range(4)), 1, 0,
                             lambda batch, job, size: job, fail,
                             steps.append, lambda: None, lambda records: records)
    assert steps == []


@pytest.mark.parametrize('tcp', [False, True])
@pytest.mark.parametrize('fail_server', [False, True])
def test_real_actor_pipeline_three_jobs_partial_group_and_abort(tmp_path, monkeypatch, tcp, fail_server):
    """Real CPU autograd/Adam/PrototypeServer; replace only CUDA hardware calls."""
    import socket
    import threading
    import time
    import torch
    import torch._dynamo
    from contextlib import nullcontext
    from types import SimpleNamespace
    from torch import nn
    from utils import sflv2, na_sfl_pipeline, na_sfl_prototype
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
    for module in (sflv2, na_sfl_pipeline):
        monkeypatch.setattr(module, '_model_device', lambda model: torch.device('cpu'))
    for module in (sflv2, na_sfl_pipeline, na_sfl_prototype):
        monkeypatch.setattr(module, '_timed', lambda device, fn: (fn(), 0.))
    cfg = load_config(overrides={'method': 'na_sfl_proto_random_sd', 'n_clients': 2,
        'slow_client_ids': [0], 'batch_size': 8, 'samples_per_client': 20,
        'random_sd': {'sample_ratio': .1, 'sample_seed': 42}, 'worker_timeout_s': 3.})
    cfg['method'] = 'na_sfl_trainer_sender_pipeline'
    cfg['client_compute_seconds'] = {'forward': .002, 'backward': .004, 'auxiliary': 0.}
    if tcp:
        cfg['network'] = {'send_buffer_sd': 3}
    server = na_sfl_prototype.PrototypeServer(nn.Sequential(nn.Conv2d(2, 10, 1),
        nn.AdaptiveAvgPool2d(1), nn.Flatten()), {**cfg, 'boundary_shape_per_sample': (2, 2, 2)})
    nodes = [na_sfl_pipeline.PipelineClient(i, nn.Conv2d(1, 2, 1), cfg,
             prototype_mode=i == 0) for i in range(2)]
    if fail_server:
        def broken(packets): raise ValueError('intentional pipeline server failure')
        server.process_batch = broken
    gates = [threading.Event() for _ in nodes]
    entered = [threading.Event() for _ in nodes]
    class GateSocket:
        def __init__(self, raw, cid): self.raw, self.cid, self.first = raw, cid, True
        def __getattr__(self, name): return getattr(self.raw, name)
        def sendall(self, data):
            if self.first:
                self.first = False
                entered[self.cid].set()
                if not gates[self.cid].wait(2):
                    raise TimeoutError('second forward did not overlap first upload')
                time.sleep(.003)
            return self.raw.sendall(data)
        def shutdown(self, how):
            gates[self.cid].set()
            return self.raw.shutdown(how)
    if tcp:
        for cid, node in enumerate(nodes):
            calls = [0]
            def forward_hook(module, inputs, cid=cid, calls=calls):
                calls[0] += 1
                if calls[0] == 2:
                    assert entered[cid].wait(1)
                    gates[cid].set()
                    time.sleep(.01)
            node.model.register_forward_pre_hook(forward_hook)
        def connections():
            result = []
            for cid in range(len(nodes)):
                client, server_socket = socket.socketpair()
                result.append((GateSocket(client, cid), server_socket))
            return result
        network = SimpleNamespace(create_connections=connections)
    else:
        network = None
    owner = SimpleNamespace(args=cfg, device='cuda:0', clients=nodes, server=server,
        isolated_augmentation=True, run_dir=tmp_path, job_records=[], network=network, _log=lambda text: None)
    def batches(images, labels, partition, round_id, cid):
        for index, size in enumerate([8, 8, 4]):
            yield index, (torch.ones(size, 1, 2, 2) * (1 + index),
                          torch.arange(size) % 2, torch.arange(index*8, index*8+size))
    owner._client_batches = batches
    server.start_round(1)
    if fail_server:
        with pytest.raises(RuntimeError, match='intentional pipeline server failure'):
            run_async_round(owner, None, None, [list(range(20))]*2, 1)
        assert owner.job_records == []
        assert all(node.pending is None for node in nodes)
        assert owner.last_async_boundary['pending_graphs'] == 0
    else:
        rows = run_async_round(owner, None, None, [list(range(20))]*2, 1)
        assert len(rows) == server.version == 6
        assert sum(row['client_optimizer_step_performed'] for row in rows) == 4
        assert [node.version for node in nodes] == [2, 2]
        assert all(node.max_pending_graphs == 2 and node.pending is None for node in nodes)
        for cid in range(2):
            own = sorted([row for row in rows if row['client_id'] == cid], key=lambda row: row['local_batch_index'])
            assert [row['client_version_before'] for row in own] == [0, 0, 1]
            assert [row['client_version_after'] for row in own] == [1, 1, 2]
            assert [row['gradient_scale_to_group_mean'] for row in own] == [.5, .5, 1.]
            assert own[0]['group_complete_s'] == own[1]['group_complete_s']
            assert own[2]['client_forward_start_s'] >= own[1]['group_complete_s']
            if tcp:
                assert min(own[1]['forward_compute_end_s'], own[0]['sd_receive_complete_s']) > max(
                    own[1]['client_forward_start_s'], own[0]['sd_send_start_s'])
            assert sum(row['cuda_time_s']['client_optimizer'] for row in own) == 0.
            assert all(row['queue_depth_at_start'] >= 1 for row in own)
        assert owner.last_async_boundary['success']
        assert owner.last_async_boundary['runtimecounts']['completed_jobs'] == 6
        if tcp:
            stats = owner.last_async_boundary['transport']
            assert stats['frames_sent'] == stats['frames_received'] == 12
            assert all(1 <= peak <= 2 for peak in stats['pipeline']['max_outstanding_jobs_per_client'])
            assert stats['pipeline']['queued_cpu_bytes'] == stats['pipeline']['outstanding_jobs'] == 0
    assert not torch.cuda.is_initialized()


def test_actual_cuda_forward_span_requires_exact_model_events_and_cpu_adapter_is_explicit():
    import torch
    from utils.deit_async import _pipeline_forward_span
    model_events, outer_events = (object(), object()), (object(), object())
    assert _pipeline_forward_span({'forward_cuda_events': model_events}, torch.device('cuda:0'), outer_events) == (model_events, 'model_forward')
    with pytest.raises(RuntimeError, match='exact model-forward CUDA events'):
        _pipeline_forward_span({'forward_cuda_events': None}, torch.device('cuda:0'), outer_events)
    assert _pipeline_forward_span({'forward_cuda_events': None}, torch.device('cpu'), outer_events) == (outer_events, 'cpu_test_adapter')

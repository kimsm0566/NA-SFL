"""Adapter lifecycle tests; no model execution or real network resources."""
import pickle
import queue
import threading
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

# Standalone collection must not rely on another test adding this path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_external' / 'FedOptima'))
from hsfl.protocol import (CancelledTraining, FrozenClientComm, FrozenServerComm,
                           StoppableScheduler, freeze_payload, recv_exact, local_pipe)
from communication.message import Message


def test_outgoing_tensor_snapshot_has_no_graph_and_keeps_required_input_grad():
    tensor = torch.tensor([1., 2.], requires_grad=True)
    frozen = freeze_payload({'x': [tensor]})['x'][0]
    with torch.no_grad():
        tensor.add_(10)
    assert frozen.tolist() == [1., 2.]
    assert frozen.is_leaf and frozen.requires_grad
    assert frozen.data_ptr() != tensor.data_ptr()


def test_queued_state_cannot_change_before_network_sender_runs():
    stop = threading.Event()
    cfg = SimpleNamespace(LABEL_ON_DEVICE=False)
    comm = FrozenClientComm(None, config=cfg, stop_event=stop)
    value = torch.tensor([1.])
    comm.send(Message(content=[{'weight': value}, None], msg_type='agg'))
    value.add_(2)
    assert comm.send_queue.get_nowait().content[0]['weight'].item() == 1


def test_cancel_releases_client_waiting_for_aggregation():
    stop = threading.Event()
    cfg = SimpleNamespace(LABEL_ON_DEVICE=False)
    comm = FrozenClientComm(None, config=cfg, stop_event=stop)
    result = []
    thread = threading.Thread(target=lambda: result.append(comm.receive(msg_type='agg')))
    thread.start()
    stop.set()
    thread.join(1)
    assert not thread.is_alive() and result == [-1]


def test_repeated_activation_permissions_cannot_build_a_slow_sender_backlog():
    comm = FrozenClientComm(None, config=SimpleNamespace(LABEL_ON_DEVICE=False),
                            stop_event=threading.Event())
    for _ in range(20):
        comm.act_avail = True  # Repeated original server ACT_CHECK permissions.
        comm.send(Message(content=[torch.ones(56, 1), torch.zeros(56,dtype=torch.long)],
                          msg_type='train'))
    assert comm.send_queue.qsize() == 1
    assert comm.sd_skipped_jobs == 19 and comm.sd_skipped_samples == 19*56
    assert comm.sd_pending == 1 and comm.sd_peak_pending == 1


def test_activation_credit_releases_after_actual_send_not_dequeue():
    comm = FrozenClientComm(object(), config=SimpleNamespace(LABEL_ON_DEVICE=False),
                            stop_event=threading.Event())
    sent = threading.Event()
    release = threading.Event()
    message = Message(content=[torch.ones(128, 1),torch.zeros(128,dtype=torch.long)],msg_type='train')
    def send(_):
        sent.set(); assert release.wait(1); return 123
    message.send_to = send
    comm.send(message)
    worker = threading.Thread(target=comm.run_sender)
    worker.start(); assert sent.wait(1)
    assert comm.sd_pending == 1
    comm.send(Message(close_comm=True))
    comm.send_queue.queue[-1].send_to = lambda _: 1
    release.set(); worker.join(1)
    assert not worker.is_alive() and comm.sd_pending == 0


def test_permission_during_busy_upload_survives_author_post_send_false():
    comm = FrozenClientComm(object(), config=SimpleNamespace(LABEL_ON_DEVICE=False),
                            stop_event=threading.Event())
    sending, release = threading.Event(), threading.Event()
    first = Message(content=[torch.ones(56,1),torch.zeros(56,dtype=torch.long)],msg_type='train')
    first.send_to = lambda _:(sending.set(),release.wait(1),123)[-1]
    comm.send(first)
    comm.act_avail = False  # Original caller's unconditional post-send line.
    sender = threading.Thread(target=comm.run_sender)
    sender.start(); assert sending.wait(1)
    comm.act_avail = True  # Last server permission arrives while upload busy.
    second = Message(content=[torch.ones(56,1),torch.zeros(56,dtype=torch.long)],msg_type='train')
    assert comm.send(second) is False
    comm.act_avail = False  # The original caller does this even for rejected SD.
    close = Message(close_comm=True); close.send_to = lambda _: 1
    comm.send(close); release.set(); sender.join(1)
    assert not sender.is_alive()
    assert comm.act_avail is True


def test_scheduler_idle_request_is_released_on_stop():
    from multiprocessing import Pipe
    compute, scheduling = Pipe()
    event = threading.Event()
    scheduler = StoppableScheduler(10, scheduling, SimpleNamespace(), 3000, stop_event=event)
    worker = threading.Thread(target=scheduler.get)
    worker.start()
    compute.send(128)
    event.set()
    assert compute.poll(1)
    kind, reason = pickle.loads(compute.recv())
    assert kind == 'hsfl_stop'
    worker.join(1)
    assert not worker.is_alive()
    compute.close(); scheduling.close()


def test_scheduler_ignores_original_epoch_validation_requests():
    scheduler = StoppableScheduler(10, None, SimpleNamespace(), 3000, stop_event=threading.Event())
    scheduler.put_data('val', Message(msg_type='val', origin=0))
    assert scheduler.val_data.empty()


def test_server_label_on_server_advances_after_one_update():
    from node.async_server_model import AsyncServerModel
    actions = []
    class Input:
        def to(self, _): return self
        def retain_grad(self): actions.append('input_grad')
    class Net:
        def train(self): actions.append('train')
        def __call__(self, _): return Input()
    class Loss:
        def item(self): return 1.0
        def backward(self): actions.append('backward')
    class Optimizer:
        def step(self): actions.append('step')
        def zero_grad(self): actions.append('zero')
    from util.timing import Timing
    model = AsyncServerModel.__new__(AsyncServerModel)
    model.net, model.device = Net(), 'unused_fake'
    model.config = SimpleNamespace(CLIENT_NUM=10, MAX_VAL_STEP=10, BATCH_SIZE=128,
        LABEL_ON_DEVICE=False, CLIP_GRAD_NORM=None, AUXILIARY='auxiliary_net')
    model.server_timing = Timing('fake')
    model.optimiser = Optimizer()
    model.criterion = lambda *args: Loss()
    model.calculate_acc = lambda *args: 10
    calls = iter([('train', [Input(), Input(), [[56, 0]]])])
    def receive(_):
        try: return next(calls)
        except StopIteration: raise CancelledTraining()
    model.receive = receive
    with pytest.raises(CancelledTraining):
        model.train_split()
    assert actions == ['train', 'input_grad', 'backward', 'step', 'zero']


def test_large_tcp_frame_short_reads_are_reassembled_and_audited():
    class Socket:
        chunks = [b'ab', b'c', b'def']
        def recv(self, remaining):
            data = self.chunks.pop(0)
            assert len(data) <= remaining
            return data
    chunks = []
    assert recv_exact(Socket(), 6, on_chunk=chunks.append) == b'abcdef'
    assert chunks == [2, 1, 3]


def test_tcp_eof_midframe_is_an_explicit_failure():
    class Socket:
        def recv(self, remaining): return b''
    with pytest.raises(ConnectionError, match='EOF'):
        recv_exact(Socket(), 4)


def test_final_large_scheduler_response_does_not_need_a_reader_to_finish():
    compute, scheduler = local_pipe()
    payload = b'x' * (2**20)
    thread = threading.Thread(target=lambda:scheduler.send(payload))
    thread.start(); thread.join(1)
    assert not thread.is_alive()
    assert compute.poll(.1) and compute.recv() == payload
    compute.close(); scheduler.close()


def test_hidden_server_receiver_failure_is_propagated_and_releases_scheduler(monkeypatch):
    comm = FrozenServerComm([object()])
    failures = []
    comm.on_receiver_error = lambda cid, error: failures.append((cid, str(error)))
    def fail(*args, **kwargs): raise ConnectionError('peer lost')
    monkeypatch.setattr('hsfl.protocol.Message', fail)
    receiver = threading.Thread(target=comm.run_receiver)
    receiver.start()
    receiver.join(1)
    assert not receiver.is_alive()
    assert failures == [(0, 'peer lost')]
    assert comm.receive_queue.get_nowait().close_comm

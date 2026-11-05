"""Small lifecycle/snapshot adapters around the author's TCP and scheduler."""
import copy
import pickle
import queue
import threading
import time
from types import SimpleNamespace

import torch

from communication.one_to_one_communicator import OneToOneCommunicator
from communication.one_to_many_communicator import OneToManyCommunicator
from communication.task_scheduler import TaskScheduler
from communication.message import Message


class CancelledTraining(Exception):
    """A requested stop, distinct from a training or communication failure."""


def recv_exact(sock, size, *, on_chunk=None):
    """TCP may split any frame, including timeout/nonblocking-backed sockets."""
    if type(size) is not int or size < 0:
        raise ValueError('frame size must be a nonnegative integer')
    result = bytearray()
    while len(result) < size:
        data = sock.recv(size-len(result))
        if not data:
            raise ConnectionError('TCP EOF inside a frame')
        if on_chunk is not None:
            on_chunk(len(data))
        result.extend(data)
    return bytes(result)


class _LocalConnection:
    """The author's compute/scheduler request protocol within one process.

    One request is outstanding. A queued response cannot block an OS pipe when
    the compute engine stops before reading a large final activation response.
    """
    def __init__(self, incoming, outgoing):
        self.incoming, self.outgoing = incoming, outgoing
        self.pending = []
        self.closed = False

    def send(self, value):
        if self.closed:
            raise BrokenPipeError('local compute channel closed')
        self.outgoing.put(value)

    def poll(self, timeout=0):
        if self.pending:
            return True
        try:
            self.pending.append(self.incoming.get(timeout=timeout))
            return True
        except queue.Empty:
            return False

    def recv(self):
        if self.closed:
            raise EOFError('local compute channel closed')
        return self.pending.pop(0) if self.pending else self.incoming.get()

    def close(self):
        self.closed = True
        self.pending.clear()
        while True:
            try: self.incoming.get_nowait()
            except queue.Empty: break


def local_pipe():
    a, b = queue.Queue(), queue.Queue()
    return _LocalConnection(a, b), _LocalConnection(b, a)


def freeze_payload(value):
    """Snapshot before enqueue, without retaining a Client autograd graph."""
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().clone()
        if value.requires_grad:
            result.requires_grad_(True)
        return result
    if isinstance(value, dict):
        return {key: freeze_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [freeze_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(freeze_payload(item) for item in value)
    return value


class FrozenClientComm(OneToOneCommunicator):
    def __init__(self, *args, stop_event, **kwargs):
        self._sd_lock = threading.Lock()
        self.sd_pending = 0
        self.sd_peak_pending = 0
        self.sd_skipped_jobs = 0
        self.sd_skipped_samples = 0
        self._act_permission = True
        super().__init__(*args, **kwargs)
        self.stop_event = stop_event

    @property
    def act_avail(self):
        with self._sd_lock:
            return self._act_permission and not self.sd_pending

    @act_avail.setter
    def act_avail(self, allowed):
        with self._sd_lock:
            if allowed:
                self._act_permission = True
            # Original AsyncClient writes False after send(). Admission consumes
            # the grant atomically in send instead: that later assignment must
            # not erase a fresh server permission received during CPU staging.

    def send(self, message):
        training = message.msg_type == 'train'
        if training:
            with self._sd_lock:
                if self.sd_pending or self.stop_event.is_set():
                    # Several ACT_CHECK replies can arrive while a slow upload
                    # is pending. Continue local Aux learning; do not enqueue
                    # redundant permissions as an unbounded SD backlog.
                    self.sd_skipped_jobs += 1
                    self.sd_skipped_samples += len(message.content[1])
                    return False
                self.sd_pending += 1
                self._act_permission = False
                self.sd_peak_pending = max(self.sd_peak_pending,self.sd_pending)
        try:
            message.content = freeze_payload(message.content)
            super().send(message)
        except BaseException:
            if training:
                with self._sd_lock: self.sd_pending -= 1
            raise
        return True

    def run_sender(self):
        while True:
            message = self.send_queue.get()
            try:
                amount = message.send_to(self.socket)
                self.amount += amount
                if message.msg_type == 'logit':
                    self.logit_amount += amount
            finally:
                if message.msg_type == 'train':
                    with self._sd_lock: self.sd_pending -= 1
            if message.close_comm:
                return

    def receive(self, msg_type=None, batch_size=None, block=True):
        if self.stop_event.is_set():
            return -1
        if not block:
            return super().receive(msg_type, batch_size, block)
        # The author's blocking receive checks only an incoming end message.
        # Poll our independent wall timer too, including during model upload.
        if msg_type in ('agg', None):
            source = self.model_queue if msg_type == 'agg' else self.receive_queue
            while not self.stop_event.is_set() and not self.stop_signal:
                try:
                    return source.get(timeout=.1)
                except queue.Empty:
                    pass
            return -1
        return super().receive(msg_type, batch_size, block)


class FrozenServerComm(OneToManyCommunicator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Local threads need neither a tensor-sharing feeder subprocess nor
        # asynchronous pickling of mutable model views. TCP framing is unchanged.
        self.send_queue = queue.Queue()
        self.receive_queue = queue.Queue()
        self.receiver_errors = []
        self.on_receiver_error = None

    def send(self, message, client_index):
        message.content = freeze_payload(message.content)
        return super().send(message, client_index)

    def run_receiver(self):
        def receive(cid):
            closed = False
            try:
                while True:
                    message = Message(from_socket=self.sockets[cid],
                                      time_difference=self.time_differences[cid])
                    message.origin = cid
                    self.receive_queue.put(message)
                    if message.close_comm:
                        closed = True
                        return
            except BaseException as error:
                self.receiver_errors.append((cid, repr(error)))
                if self.on_receiver_error is not None:
                    self.on_receiver_error(cid, error)
            finally:
                if not closed:
                    # Release scheduler.put's close counter on failed peers too.
                    self.receive_queue.put(SimpleNamespace(close_comm=True, origin=cid))
        workers = [threading.Thread(target=receive, args=(cid,), daemon=True,
                   name='fedoptima-server-input-%d'%cid) for cid in range(self.client_num)]
        for worker in workers: worker.start()
        for worker in workers: worker.join()


class StoppableScheduler(TaskScheduler):
    def __init__(self, *args, stop_event, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.RLock()
        self.stop_event = stop_event
        self.received_samples = [0] * self.client_num
        self.selected_samples = [0] * self.client_num
        self.received_jobs = [0] * self.client_num
        self.peak_buffered_samples = 0

    def put_data(self, message_type, message=None):
        # Evaluation is globally triggered by completed visits, not asynchronous
        # local epoch messages (which otherwise bias the stop window).
        if message_type == 'val':
            return
        with self.lock:
            if message_type == 'train':
                self.received_samples[message.origin] += int(message.content[0].shape[0])
                self.received_jobs[message.origin] += 1
            super().put_data(message_type, message)
            if message_type == 'train':
                self.peak_buffered_samples = max(self.peak_buffered_samples,
                    sum(int(x.shape[0]) for x in self.client_act_queues if x is not None))

    def get(self):
        while True:
            if not self.connection.poll(.1):
                if self.stop_event.is_set():
                    return
                continue
            size = self.connection.recv()
            if size == -1:
                return
            while not self.stop_event.is_set():
                kind, content = self.get_data(size)
                if kind != 'wait':
                    if kind == 'train':
                        for samples, origin in content[2]:
                            self.selected_samples[origin] += int(samples)
                    self.connection.send(pickle.dumps([kind, content]))
                    break
                self.stop_event.wait(.002)
            else:
                self.connection.send(pickle.dumps(['hsfl_stop', 'requested_stop']))
                return

    def counters(self):
        with self.lock:
            return dict(received_samples=list(self.received_samples),
                selected_samples=list(self.selected_samples),
                received_jobs=list(self.received_jobs),
                buffered_samples=[0 if x is None else int(x.shape[0]) for x in self.client_act_queues],
                peak_buffered_samples=self.peak_buffered_samples)


class WireAudit:
    """Actual framed TCP bytes, counted at both ends of each namespace pair."""
    def __init__(self, pairs):
        self.roles = {}
        self._lock = threading.Lock()
        self.counts = {cid: {direction: {'sent': 0, 'received': 0, 'messages': {}}
                      for direction in ('upload', 'download')} for cid in range(len(pairs))}
        for cid, (client, server) in enumerate(pairs):
            self.roles[client.fileno()] = (cid, 'upload')
            self.roles[server.fileno()] = (cid, 'download')

    def sent(self, sock, length, kind):
        role = self.roles.get(sock.fileno())
        if role is None:
            return
        cid, direction = role
        with self._lock:
            item = self.counts[cid][direction]
            item['sent'] += length
            item['messages'][kind] = item['messages'].get(kind, 0) + 1

    def received(self, sock, length):
        role = self.roles.get(sock.fileno())
        if role is None:
            return
        cid, sending = role
        direction = 'download' if sending == 'upload' else 'upload'
        with self._lock:
            self.counts[cid][direction]['received'] += length

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self.counts)

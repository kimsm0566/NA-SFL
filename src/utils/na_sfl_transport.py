"""Bounded CPU tensor frames and per-Client duplex TCP round workers.

The wire format is an eight-byte network-order JSON length, strict UTF-8 JSON,
then contiguous native little-endian tensor payloads. It never uses pickle.
Framing helpers leave socket timeout/lifecycle ownership to their caller.
"""
import hashlib
import json
import math
import socket
import struct
import sys
import threading
import time
from collections import deque

import torch

MAX_HEADER_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
MAX_TENSORS = 4096
_DTYPES = {name: getattr(torch, name) for name in (
    'float16', 'bfloat16', 'float32', 'float64', 'int8', 'uint8',
    'int16', 'int32', 'int64', 'bool')}
_ITEM_SIZES = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'float64': 8,
               'int8': 1, 'uint8': 1, 'int16': 2, 'int32': 4, 'int64': 8, 'bool': 1}


def _integer(value):
    return type(value) is int


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def _json_bytes(value):
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True,
                          separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError('metadata must be finite JSON') from error


def _raw_tensor(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
        raise ValueError('transport requires CPU tensors')
    if tensor.layout != torch.strided or str(tensor.dtype).removeprefix('torch.') not in _DTYPES:
        raise ValueError('unsupported tensor dtype/layout')
    if sys.byteorder != 'little':
        raise ValueError('tensor wire format requires a little-endian host')
    # uint8 view also supports bfloat16 and scalar integer state buffers.
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _descriptor_size(spec):
    if not isinstance(spec, dict) or set(spec) != {'name', 'dtype', 'shape', 'nbytes', 'sha256'}:
        raise ValueError('invalid tensor descriptor')
    if not isinstance(spec['name'], str) or not spec['name'] or len(spec['name']) > 1024:
        raise ValueError('invalid tensor name')
    dtype, shape = spec['dtype'], spec['shape']
    if not isinstance(dtype, str) or dtype not in _DTYPES:
        raise ValueError('unsupported tensor dtype')
    if not isinstance(shape, list) or len(shape) > 16:
        raise ValueError('invalid tensor shape')
    count = 1
    for dimension in shape:
        if not _integer(dimension) or not 0 <= dimension <= 2 ** 31 - 1:
            raise ValueError('invalid tensor dimension')
        count *= dimension
        if count > MAX_PAYLOAD_BYTES:
            raise ValueError('tensor is too large')
    size = count * _ITEM_SIZES[dtype]
    if not _integer(spec['nbytes']) or spec['nbytes'] != size or size > MAX_PAYLOAD_BYTES:
        raise ValueError('tensor byte size does not match shape/dtype')
    digest = spec['sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid SHA256')
    return size


def send_tensors(sock, tensors, metadata):
    """Send a checked frame; return payload/header/wire byte and tensor counts."""
    if not isinstance(tensors, dict) or len(tensors) > MAX_TENSORS or not isinstance(metadata, dict):
        raise ValueError('frame requires tensor and metadata dictionaries')
    payloads, specs, total = [], [], 0
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError('transport requires tensors')
        # Check bounds before materializing a possibly huge byte copy.
        size = tensor.numel() * tensor.element_size()
        total += size
        if total > MAX_PAYLOAD_BYTES:
            raise ValueError('frame payload is too large')
        raw = _raw_tensor(tensor)
        spec = {'name': name, 'dtype': str(tensor.dtype).removeprefix('torch.'),
                'shape': list(tensor.shape), 'nbytes': size,
                'sha256': hashlib.sha256(raw).hexdigest()}
        _descriptor_size(spec)
        specs.append(spec)
        payloads.append(raw)
    encoded = _json_bytes({'version': 1, 'metadata': metadata,
                           'tensors': specs, 'payload_bytes': total})
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError('frame header is too large')
    sock.sendall(struct.pack('!Q', len(encoded)))
    sock.sendall(encoded)
    for raw in payloads:
        if raw:
            sock.sendall(raw)
    return {'payload_bytes': total, 'header_bytes': len(encoded),
            'wire_bytes': 8 + len(encoded) + total, 'tensor_count': len(specs)}


def _read_exact(sock, size):
    raw = bytearray(size)
    view = memoryview(raw)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:])
        if not received:
            raise EOFError('EOF in tensor frame')
        offset += received
    return raw


def recv_tensors(sock, on_payload_received=None):
    """Receive and validate the entire frame before exposing any CPU tensor."""
    length = struct.unpack('!Q', _read_exact(sock, 8))[0]
    if not 0 < length <= MAX_HEADER_BYTES:
        raise ValueError('invalid frame header length')
    try:
        header = json.loads(_read_exact(sock, length).decode('utf-8'),
                            object_pairs_hook=_strict_object, parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('invalid frame JSON') from error
    if not isinstance(header, dict) or set(header) != {'version', 'metadata', 'tensors', 'payload_bytes'}:
        raise ValueError('invalid frame envelope')
    if type(header['version']) is not int or header['version'] != 1 or not isinstance(header['metadata'], dict):
        raise ValueError('invalid frame version/metadata')
    # json.loads may overflow a valid JSON exponent to infinity.
    _json_bytes(header['metadata'])
    specs = header['tensors']
    if not isinstance(specs, list) or len(specs) > MAX_TENSORS:
        raise ValueError('invalid tensor count')
    total, names = 0, set()
    for spec in specs:
        total += _descriptor_size(spec)
        if spec['name'] in names or total > MAX_PAYLOAD_BYTES:
            raise ValueError('duplicate tensor or excessive payload')
        names.add(spec['name'])
    if not _integer(header['payload_bytes']) or total != header['payload_bytes']:
        raise ValueError('invalid total payload size')
    tensors = {}
    for index, spec in enumerate(specs):
        raw = _read_exact(sock, spec['nbytes'])
        if index == len(specs) - 1 and on_payload_received is not None:
            on_payload_received(total)
        if hashlib.sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('tensor SHA256 mismatch')
        dtype = _DTYPES[spec['dtype']]
        tensor = (torch.frombuffer(raw, dtype=dtype).clone() if raw else torch.empty(0, dtype=dtype))
        tensors[spec['name']] = tensor.reshape(spec['shape'])
    if not specs and on_payload_received is not None:
        on_payload_received(0)
    return tensors, header['metadata']


class TCPRoundTransport:
    """Four CPU workers per Client on an already connected duplex socket pair.

    ``reserve(cid, check_abort)`` owns one credit in the calling Client thread;
    ``submit(packet)`` consumes that reservation. Credit spans reservation,
    queued/transmitting bytes, and the fully decoded SD's ``on_sd`` admission.
    ``on_sd`` must return after admission and must itself respond to the owning
    round's cancellation if it blocks on a full Server queue. ``ready``,
    ``future`` and the exact shared ``row`` stay local; only tensors and job
    identity traverse TCP. The Server must call ``send_gradient`` once per S
    job. A jobs never have a download. No worker calls CUDA.

    finish() drains all submitted uploads, all required S gradients and their
    callbacks, joins workers, then closes owned sockets. abort() wakes credits
    and interrupts socket I/O; errors invoke on_error exactly once. Stats are
    JSON-compatible counters and frame events, with pending_sd including held
    reservations and pending_gradients counting S jobs until DL callback ends.
    """

    def __init__(self, connections, on_sd, on_gradient, on_error, now,
                 capacity=3, worker_timeout_s=120.):
        if not isinstance(connections, (list, tuple)) or not connections:
            raise ValueError('connections must contain one socket pair per Client')
        if not _integer(capacity) or capacity < 1:
            raise ValueError('SD capacity must be a positive integer')
        if isinstance(worker_timeout_s, bool) or not isinstance(worker_timeout_s, (int, float)) or not math.isfinite(worker_timeout_s) or worker_timeout_s <= 0:
            raise ValueError('worker timeout must be finite and positive')
        if any(not callable(value) for value in (on_sd, on_gradient, on_error, now)):
            raise ValueError('transport callbacks and clock must be callable')
        if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in connections):
            raise ValueError('each connection requires Client and Server sockets')
        self.connections = list(connections)
        self.on_sd, self.on_gradient = on_sd, on_gradient
        self.on_error, self.now = on_error, now
        self.capacity, self.worker_timeout_s = capacity, float(worker_timeout_s)
        self._condition = threading.Condition()
        self._started = self._closing = self._aborted = self._finished = False
        self._error = None
        count = len(connections)
        self._credits = [0] * count
        self._max_credits = [0] * count
        self._reserved = {}
        self._records = {}
        self._seen_jobs = set()
        self._sd_pending = self._gradient_pending = 0
        self._sd_arrival_next = self._sd_admission_next = 0
        self._queues = {kind: [deque() for _ in connections]
                        for kind in ('sd_send', 'sd_receive', 'gradient_send', 'gradient_receive')}
        self._threads = []
        self._events = []
        self._counters = {f'{kind}_payload_bytes_{direction}': 0
                          for kind in ('sd', 'gradient') for direction in ('sent', 'received')}
        self._counters.update(frames_sent=0, frames_received=0, wire_bytes_sent=0)

    def start(self):
        with self._condition:
            if self._started or self._aborted or self._finished:
                raise RuntimeError('transport cannot be started twice')
            self._started = True
        try:
            for pair in self.connections:
                for sock in pair:
                    sock.settimeout(self.worker_timeout_s)
            for cid in range(len(self.connections)):
                for kind in self._queues:
                    thread = threading.Thread(target=self._worker, args=(cid, kind),
                                              name=f'na-tcp-{cid}-{kind}', daemon=True)
                    self._threads.append(thread)
                    thread.start()
        except BaseException as error:
            self.abort(error)
            raise
        return self

    def _check_running(self):
        if self._aborted:
            raise RuntimeError('TCP round transport aborted') from self._error
        if not self._started or self._closing or self._finished:
            raise RuntimeError('TCP round transport is not accepting work')

    def _client_id(self, cid):
        if not _integer(cid) or not 0 <= cid < len(self.connections):
            raise ValueError('invalid Client id')
        return cid

    def reserve(self, cid, check_abort):
        """Acquire bounded SD storage before the calling Client's forward."""
        cid = self._client_id(cid)
        if not callable(check_abort):
            raise ValueError('check_abort must be callable')
        with self._condition:
            while True:
                self._check_running()
                if check_abort():
                    raise RuntimeError('owning round aborted')
                if self._credits[cid] < self.capacity:
                    self._credits[cid] += 1
                    self._max_credits[cid] = max(self._max_credits[cid], self._credits[cid])
                    key = (cid, threading.get_ident())
                    self._reserved[key] = self._reserved.get(key, 0) + 1
                    return
                self._condition.wait(.05)

    def _identity(self, packet):
        if not isinstance(packet, dict) or not isinstance(packet.get('row'), dict):
            raise ValueError('packet requires a shared row dictionary')
        row = packet['row']
        cid = self._client_id(row.get('client_id'))
        job_id = row.get('job_id')
        if not isinstance(job_id, (list, tuple)) or len(job_id) != 3 or any(not _integer(v) or v < 0 for v in job_id):
            raise ValueError('job_id must be (round, Client, batch) nonnegative integers')
        if job_id[1] != cid:
            raise ValueError('job Client does not match connection')
        return cid, tuple(job_id)

    def _snapshot(self, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.layout != torch.strided:
            raise ValueError('round transport requires dense CPU tensors')
        if str(tensor.dtype).removeprefix('torch.') not in _DTYPES:
            raise ValueError('unsupported CPU tensor dtype')
        if tensor.numel() * tensor.element_size() > MAX_PAYLOAD_BYTES:
            raise ValueError('tensor is too large')
        return tensor.detach().clone(memory_format=torch.contiguous_format)

    def submit(self, packet):
        """Snapshot and enqueue SD; actual send never runs in the Client thread."""
        cid, job_id = self._identity(packet)
        returns = packet.get('return_cut_gradient')
        if type(returns) is not bool:
            raise ValueError('return_cut_gradient must be boolean')
        # Reserving is required before even creating another potentially large
        # tensor snapshot. Ownership prevents one producer taking another's slot.
        key = (cid, threading.get_ident())
        with self._condition:
            self._check_running()
            if self._reserved.get(key, 0) < 1:
                raise RuntimeError('submit requires a credit reserved by this Client thread')
            if job_id in self._seen_jobs:
                raise ValueError('duplicate SD job')
        activation = self._snapshot(packet.get('activation'))
        labels = self._snapshot(packet.get('labels'))
        if activation.ndim < 1 or labels.ndim != 1 or labels.dtype != torch.int64 or activation.shape[0] != labels.shape[0]:
            raise ValueError('SD activation/labels sample shape mismatch')
        total = activation.numel() * activation.element_size() + labels.numel() * labels.element_size()
        if total > MAX_PAYLOAD_BYTES:
            raise ValueError('SD frame is too large')
        record = {'packet': dict(packet), 'row': packet['row'], 'cid': cid,
                  'job_id': job_id, 'returns': returns,
                  'sd': {'activation': activation, 'labels': labels},
                  'gradient_queued': False, 'activation_shape': activation.shape,
                  'activation_dtype': activation.dtype}
        record['packet']['row'] = packet['row']
        # Drop source tensor references: updates by the caller cannot alter SD.
        record['packet']['activation'] = activation
        record['packet']['labels'] = labels
        if packet.get('job_kind') in ('SD_JOB', 'PROTO_JOB'):
            from utils.na_sfl_proto_protocol import request_message
            if packet['job_kind'] == 'PROTO_JOB':
                record['packet']['counts'] = self._snapshot(packet.get('counts'))
            record['typed_request'], record['typed_metadata'] = request_message(record['packet'])
            record['row']['request_kind'] = packet['job_kind']
        with self._condition:
            self._check_running()
            self._reserved[key] -= 1
            if not self._reserved[key]:
                del self._reserved[key]
            self._records[job_id] = record
            self._seen_jobs.add(job_id)
            self._sd_pending += 1
            self._gradient_pending += int(returns)
            self._queues['sd_send'][cid].append(record)
            self._queues['sd_receive'][cid].append(record)
            self._condition.notify_all()

    def send_gradient(self, packet, gradient_CPU):
        """Snapshot a cut gradient and enqueue DL without waiting for TCP."""
        cid, job_id = self._identity(packet)
        with self._condition:
            self._check_running()
            record = self._records.get(job_id)
            if record is None or not record['returns']:
                raise ValueError('no cut gradient is allowed for this job')
            if record['gradient_queued']:
                raise ValueError('duplicate cut gradient')
            # Existing Clients are in-flight1. Bound DL even if a caller violates
            # that contract, rather than building an unlimited Server backlog.
            in_flight = sum(1 for value in self._records.values()
                            if value['cid'] == cid and value['gradient_queued'])
            if in_flight >= self.capacity:
                raise RuntimeError('bounded gradient queue is full')
        gradient = self._snapshot(gradient_CPU)
        if gradient.shape != record['activation_shape'] or gradient.dtype != record['activation_dtype']:
            raise ValueError('cut gradient shape/dtype mismatch')
        with self._condition:
            self._check_running()
            if record['gradient_queued']:
                raise ValueError('duplicate cut gradient')
            record['gradient_queued'] = True
            record['gradient'] = gradient
            if 'typed_metadata' in record:
                from utils.na_sfl_proto_protocol import reply_message
                record['typed_reply'], record['reply_metadata'] = reply_message(
                    record['typed_metadata'], gradient, record['packet']['labels'])
                record['row']['response_kind'] = record['reply_metadata']['kind']
            self._queues['gradient_send'][cid].append(record)
            self._queues['gradient_receive'][cid].append(record)
            self._condition.notify_all()

    def _next(self, cid, kind):
        with self._condition:
            queue = self._queues[kind][cid]
            while not queue and not self._closing and not self._aborted:
                self._condition.wait(.05)
            if self._aborted or not queue:
                return None
            return queue.popleft()

    def _metadata(self, record, kind):
        if 'typed_metadata' in record:
            return record['typed_metadata'] if kind == 'sd' else record['reply_metadata']
        return {'kind': kind, 'job_id': list(record['job_id']),
                'client_id': record['cid'], 'return_cut_gradient': record['returns']}

    def _event(self, record, kind, phase, payload_bytes=None):
        stamp = self.now()
        prefix = 'sd' if kind == 'sd' else 'gradient'
        with self._condition:
            record['row'][f'{prefix}_{phase}_s'] = stamp
            event = {'kind': kind, 'phase': phase, 'time_s': stamp,
                     'job_id': list(record['job_id']), 'client_id': record['cid']}
            if payload_bytes is not None:
                event['payload_bytes'] = payload_bytes
            self._events.append(event)

    def _send(self, record, kind, sock):
        tensors = record['sd'] if kind == 'sd' else {'gradient': record['gradient']}
        if 'typed_metadata' in record:
            tensors = record['typed_request'] if kind == 'sd' else record['typed_reply']
        self._event(record, kind, 'send_start')
        stats = send_tensors(sock, tensors, self._metadata(record, kind))
        record['row'][kind + '_payload_bytes'] = stats['payload_bytes']
        record['row'][kind + '_header_bytes'] = stats['header_bytes']
        record['row'][kind + '_wire_bytes'] = stats['wire_bytes']
        self._event(record, kind, 'send_return', stats['payload_bytes'])
        with self._condition:
            self._counters[f'{kind}_payload_bytes_sent'] += stats['payload_bytes']
            self._counters['frames_sent'] += 1
            self._counters['wire_bytes_sent'] += stats['wire_bytes']
            self._condition.notify_all()

    def _receive(self, record, kind, sock):
        ticket = None
        def received(payload_bytes):
            nonlocal ticket
            # Serialize the full-frame receipt stamp and ticket, then validate
            # outside the lock. A later small frame must not overtake an earlier
            # frame merely because its integrity checks finish sooner.
            with self._condition:
                self._event(record, kind, 'receive_complete', payload_bytes)
                if kind == 'sd':
                    ticket = self._sd_arrival_next
                    self._sd_arrival_next += 1
                    record['row']['transport_arrival_seq'] = ticket
        tensors, metadata = recv_tensors(sock, on_payload_received=received)
        if metadata != self._metadata(record, kind):
            raise ValueError('received frame job identity does not match expected job')
        expected_keys = {'activation', 'labels'} if kind == 'sd' else {'gradient'}
        if 'typed_metadata' in record:
            expected_keys = set(record['typed_request'] if kind == 'sd' else record['typed_reply'])
        if set(tensors) != expected_keys:
            raise ValueError('received frame tensor names do not match job type')
        expected = record['sd'] if kind == 'sd' else {'gradient': record['gradient']}
        if 'typed_metadata' in record:
            from utils.na_sfl_proto_protocol import validate_message
            validate_message(tensors, metadata)
            expected = record['typed_request'] if kind == 'sd' else record['typed_reply']
        if any(tensors[key].shape != value.shape or tensors[key].dtype != value.dtype for key, value in expected.items()):
            raise ValueError('received frame tensor shape/dtype mismatch')
        # A self-consistent but misrouted/replaced frame must also match the
        # immutable originating job, including NaN bit patterns and signed zero.
        if any(_raw_tensor(tensors[key]) != _raw_tensor(value) for key, value in expected.items()):
            raise ValueError('received tensor bytes do not match originating job')
        payload_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        record['row'][kind + '_validation_complete_s'] = self.now()
        with self._condition:
            self._counters[f'{kind}_payload_bytes_received'] += payload_bytes
            self._counters['frames_received'] += 1
        if kind == 'sd':
            decoded = dict(record['packet'])
            if 'typed_metadata' in record:
                if metadata['kind'] == 'SD_JOB':
                    decoded.update(activation=tensors['sd'], labels=tensors['labels'])
                else:
                    decoded.update(activation=tensors['prototypes'], labels=tensors['classes'], counts=tensors['counts'])
            else:
                decoded.update(tensors)
            with self._condition:
                while ticket != self._sd_admission_next and not self._aborted:
                    self._condition.wait(.05)
                if self._aborted:
                    return
            self.on_sd(decoded)
            with self._condition:
                if self._aborted:
                    return
                self._sd_admission_next += 1
                # Admission, not send return or Server processing, releases SD.
                self._credits[record['cid']] -= 1
                self._sd_pending -= 1
                if not record['returns']:
                    self._records.pop(record['job_id'], None)
                self._condition.notify_all()
        else:
            self.on_gradient(record['job_id'], tensors['gradient'])
            with self._condition:
                if self._aborted:
                    return
                self._gradient_pending -= 1
                self._records.pop(record['job_id'], None)
                self._condition.notify_all()

    def _worker(self, cid, kind):
        try:
            while True:
                record = self._next(cid, kind)
                if record is None:
                    return
                frame_kind = 'sd' if kind.startswith('sd') else 'gradient'
                client, server = self.connections[cid]
                if kind.endswith('send'):
                    self._send(record, frame_kind, client if frame_kind == 'sd' else server)
                else:
                    self._receive(record, frame_kind, server if frame_kind == 'sd' else client)
        except BaseException as error:
            self.abort(error)

    def _close_sockets(self):
        for pair in self.connections:
            for sock in pair:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass

    def abort(self, error=None):
        """Cancel once, release all outstanding credits, and interrupt I/O."""
        with self._condition:
            if self._aborted or self._finished:
                return
            self._aborted = True
            self._error = error
            self._credits[:] = [0] * len(self.connections)
            self._reserved.clear()
            self._records.clear()
            self._sd_pending = self._gradient_pending = 0
            for queues in self._queues.values():
                for queue in queues:
                    queue.clear()
            self._condition.notify_all()
        self._close_sockets()
        if error is not None:
            # The original failure remains the abort cause if reporting itself
            # raises; no recursive error callback or second notification occurs.
            try:
                self.on_error(error)
            except BaseException:
                pass

    def finish(self):
        """Drain uploads and required downloads, join workers and close sockets."""
        deadline = time.monotonic() + self.worker_timeout_s
        try:
            with self._condition:
                if self._finished:
                    return
                if not self._started:
                    raise RuntimeError('transport has not started')
                while not self._aborted and (self._sd_pending or self._gradient_pending or self._reserved):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('TCP round did not drain before timeout')
                    self._condition.wait(min(.05, remaining))
                self._closing = True
                self._condition.notify_all()
            for thread in self._threads:
                if thread is threading.current_thread():
                    raise RuntimeError('finish cannot be called in a transport callback')
                thread.join(max(0., deadline - time.monotonic()))
            if any(thread.is_alive() for thread in self._threads):
                raise TimeoutError('TCP workers did not exit before timeout')
            with self._condition:
                if self._aborted:
                    raise RuntimeError('TCP round transport aborted') from self._error
                self._finished = True
            self._close_sockets()
        except BaseException as error:
            self.abort(error)
            # The drain deadline may have expired. Cancellation wakes the
            # queues/socket I/O; give those workers a separate finite join
            # window before the caller handles the original failure.
            cleanup_deadline = time.monotonic() + self.worker_timeout_s
            for thread in self._threads:
                if thread is not threading.current_thread():
                    thread.join(max(0., cleanup_deadline - time.monotonic()))
            raise

    def stats(self):
        """Return a detached JSON-compatible round communication snapshot."""
        with self._condition:
            return {**self._counters, 'workers_alive': sum(t.is_alive() for t in self._threads),
                    'pending_sd': sum(self._credits), 'pending_gradients': self._gradient_pending,
                    'reserved_sd': sum(self._reserved.values()),
                    'max_pending_sd_per_client': list(self._max_credits),
                    'frame_events': [dict(event) for event in self._events],
                    'aborted': self._aborted, 'finished': self._finished,
                    'error': str(self._error) if self._error is not None else None}

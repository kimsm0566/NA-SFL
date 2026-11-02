"""CPU protocol/lifecycle checks; these do not execute model forward/backward."""
import hashlib
import json
import socket
import struct
import threading
import time

import pytest
import torch


def api():
    from utils import na_sfl_transport
    return na_sfl_transport


def inet_pair():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname(), timeout=2)
    server, _ = listener.accept()
    listener.close()
    return client, server


@pytest.mark.parametrize('factory', [socket.socketpair, inet_pair])
def test_frame_preserves_noncontiguous_state_and_counts_real_payload(factory):
    transport = api()
    client, server = factory()
    try:
        client.settimeout(2)
        server.settimeout(2)
        source = {'weight': torch.arange(12, dtype=torch.float32).reshape(3, 4).T,
                  'bn.num_batches_tracked': torch.tensor(7, dtype=torch.int64),
                  'empty': torch.empty(0, dtype=torch.float32)}
        stats = transport.send_tensors(client, source, {'round': 2, 'purpose': 'broadcast'})
        tensors, metadata = transport.recv_tensors(server)
        assert metadata == {'round': 2, 'purpose': 'broadcast'}
        assert stats['payload_bytes'] == 56
        assert stats['wire_bytes'] == stats['header_bytes'] + 56 + 8
        assert stats['tensor_count'] == 3
        assert all(torch.equal(tensors[k], v) for k, v in source.items())
        assert all(v.device.type == 'cpu' and v.is_contiguous() for v in tensors.values())
        tensors['weight'].zero_()
        assert source['weight'][0, 1].item() == 4
    finally:
        client.close()
        server.close()


def manual_frame(spec=None, payload=b'\x00\x00\x80?'):
    if spec is None:
        spec = {'name': 'x', 'dtype': 'float32', 'shape': [1], 'nbytes': 4,
                'sha256': hashlib.sha256(payload).hexdigest()}
    header = {'version': 1, 'metadata': {}, 'tensors': [spec], 'payload_bytes': 4}
    encoded = json.dumps(header, separators=(',', ':')).encode()
    return struct.pack('!Q', len(encoded)) + encoded + payload


@pytest.mark.parametrize('mutation', ['hash', 'size', 'shape', 'dtype', 'name'])
def test_frame_rejects_corruption_and_invalid_tensor_descriptors(mutation):
    transport = api()
    spec = {'name': 'x', 'dtype': 'float32', 'shape': [1], 'nbytes': 4,
            'sha256': hashlib.sha256(b'\x00\x00\x80?').hexdigest()}
    spec.update({'hash': {'sha256': '0' * 64}, 'size': {'nbytes': 8},
                 'shape': {'shape': [True]}, 'dtype': {'dtype': 'object'},
                 'name': {'name': ''}}[mutation])
    client, server = socket.socketpair()
    try:
        client.sendall(manual_frame(spec))
        with pytest.raises(ValueError):
            transport.recv_tensors(server)
    finally:
        client.close()
        server.close()


def test_frame_bounds_header_before_allocation_and_detects_eof():
    transport = api()
    client, server = socket.socketpair()
    try:
        client.sendall(struct.pack('!Q', 2 ** 40))
        with pytest.raises(ValueError):
            transport.recv_tensors(server)
        client.sendall(struct.pack('!Q', 100) + b'{')
        client.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError):
            transport.recv_tensors(server)
    finally:
        client.close()
        server.close()


def test_frame_rejects_nonfinite_metadata_before_sending():
    transport = api()
    client, server = socket.socketpair()
    try:
        with pytest.raises(ValueError):
            transport.send_tensors(client, {'x': torch.ones(1)}, {'bad': float('nan')})
        server.settimeout(.03)
        with pytest.raises(socket.timeout):
            server.recv(1)
    finally:
        client.close()
        server.close()


class PausedSocket:
    """Gate a real socket operation to control when TCP bytes can progress."""
    def __init__(self, raw, operation):
        self.raw, self.operation = raw, operation
        self.entered, self.release = threading.Event(), threading.Event()

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def _pause(self, operation):
        if operation == self.operation:
            self.entered.set()
            if not self.release.wait(3):
                raise TimeoutError('test socket gate was not released')

    def sendall(self, data):
        self._pause('send')
        return self.raw.sendall(data)

    def recv_into(self, data):
        self._pause('receive')
        return self.raw.recv_into(data)

    def shutdown(self, how):
        self.release.set()
        return self.raw.shutdown(how)


def packet(index=0, returns=False, size=2):
    return {'row': {'job_id': (1, 0, index), 'client_id': 0},
            'activation': torch.arange(size * 2, dtype=torch.float32).reshape(size, 2),
            'labels': torch.arange(size, dtype=torch.int64),
            'future': object() if returns else None, 'ready': object(),
            'return_cut_gradient': returns}


def make_round(connections, on_sd, on_gradient=lambda *_: None, capacity=3, errors=None):
    origin = time.monotonic()
    return api().TCPRoundTransport(connections, on_sd, on_gradient,
                                    (errors if errors is not None else []).append,
                                    lambda: time.monotonic() - origin,
                                    capacity=capacity, worker_timeout_s=2).start()


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), 'condition did not become true before deadline'


def test_round_actual_tcp_preserves_local_row_future_and_immutable_sd():
    client, server = inet_pair()
    blocked_client = PausedSocket(client, 'send')
    accepted, errors = [], []
    round_transport = make_round([(blocked_client, server)], accepted.append, errors=errors)
    item = packet()
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(item)
        assert blocked_client.entered.wait(1)
        # This local work runs while the upload worker is still blocked in send.
        item['activation'].fill_(999)
        item['labels'].fill_(9)
        assert accepted == []
        blocked_client.release.set()
        wait_until(lambda: len(accepted) == 1)
        decoded = accepted[0]
        assert decoded['row'] is item['row']
        assert decoded['future'] is item['future'] and decoded['ready'] is item['ready']
        assert torch.equal(decoded['activation'], torch.tensor([[0., 1.], [2., 3.]]))
        assert torch.equal(decoded['labels'], torch.tensor([0, 1]))
        round_transport.finish()
        stats = round_transport.stats()
        assert stats['sd_payload_bytes_sent'] == stats['sd_payload_bytes_received'] == 32
        assert stats['gradient_payload_bytes_sent'] == stats['gradient_payload_bytes_received'] == 0
        assert stats['pending_sd'] == stats['pending_gradients'] == stats['workers_alive'] == 0
        assert len(stats['frame_events']) == 3
        assert errors == []
        assert all(key in item['row'] for key in
                   ['sd_send_start_s', 'sd_send_return_s', 'sd_receive_complete_s'])
        assert client.fileno() == server.fileno() == -1
        json.dumps(stats, allow_nan=False)
    finally:
        blocked_client.release.set()
        round_transport.abort()


def test_sd_credit_blocks_before_compute_until_complete_receiver_admission():
    accepted = threading.Event()
    admit = threading.Event()
    errors = []

    def on_sd(_):
        accepted.set()
        assert admit.wait(2)

    round_transport = make_round([socket.socketpair()], on_sd, capacity=1, errors=errors)
    producer_reached_compute = threading.Event()
    producer_errors = []

    def next_compute():
        try:
            round_transport.reserve(0, lambda: None)
            producer_reached_compute.set()
            round_transport.submit(packet(1))
        except BaseException as error:
            producer_errors.append(error)

    producer = threading.Thread(target=next_compute)
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(packet())
        assert accepted.wait(1)
        wait_until(lambda: round_transport.stats()['sd_payload_bytes_sent'] == 32)
        producer.start()
        assert not producer_reached_compute.wait(.05)
        assert round_transport.stats()['pending_sd'] == 1
        admit.set()
        assert producer_reached_compute.wait(1)
        producer.join(1)
        round_transport.finish()
        assert producer_errors == errors == []
        assert round_transport.stats()['max_pending_sd_per_client'] == [1]
    finally:
        admit.set()
        round_transport.abort()
        if producer.ident is not None:
            producer.join(1)


def test_gradient_worker_never_blocks_server_and_callback_waits_full_download():
    client, server = socket.socketpair()
    gated_server = PausedSocket(server, 'send')
    gated_client = PausedSocket(client, 'receive')
    accepted, received, errors = [], [], []
    round_transport = make_round([(gated_client, gated_server)], accepted.append,
                                 lambda job, tensor: received.append((job, tensor)), errors=errors)
    item = packet(returns=True)
    gradient = torch.full((2, 2), 4., dtype=torch.float32)
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(item)
        wait_until(lambda: len(accepted) == 1)
        round_transport.send_gradient(accepted[0], gradient)
        assert gated_server.entered.wait(1)
        gradient.zero_()  # The queued download owns its immutable snapshot.
        assert received == []
        gated_server.release.set()
        assert gated_client.entered.wait(1)
        wait_until(lambda: 'gradient_send_return_s' in item['row'])
        assert received == []
        assert 'gradient_receive_complete_s' not in item['row']
        gated_client.release.set()
        wait_until(lambda: len(received) == 1)
        assert received[0][0] == (1, 0, 0)
        assert torch.equal(received[0][1], torch.full((2, 2), 4.))
        round_transport.finish()
        assert errors == []
        assert round_transport.stats()['gradient_payload_bytes_received'] == 16
        assert len(round_transport.stats()['frame_events']) == 6
    finally:
        gated_client.release.set()
        gated_server.release.set()
        round_transport.abort()


def test_a_packets_forbid_gradient_and_finish_drains_pending_uploads():
    client, server = socket.socketpair()
    gated = PausedSocket(client, 'send')
    accepted, downloaded = [], []
    round_transport = make_round([(gated, server)], accepted.append,
                                 lambda *_: downloaded.append(True))
    item = packet()
    finished, finish_errors = threading.Event(), []

    def finish():
        try:
            round_transport.finish()
            finished.set()
        except BaseException as error:
            finish_errors.append(error)

    finalizer = threading.Thread(target=finish)
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(item)
        assert gated.entered.wait(1)
        with pytest.raises(ValueError):
            round_transport.send_gradient(item, torch.ones(2, 2))
        finalizer.start()
        assert not finished.wait(.05)
        gated.release.set()
        assert finished.wait(1)
        finalizer.join(1)
        assert finish_errors == downloaded == [] and len(accepted) == 1
        assert round_transport.stats()['workers_alive'] == 0
    finally:
        gated.release.set()
        round_transport.abort()
        if finalizer.ident is not None:
            finalizer.join(1)


def test_peer_failure_reports_once_closes_all_and_wakes_reserved_producer():
    client, server = socket.socketpair()
    gated = PausedSocket(client, 'send')
    errors, producer_errors = [], []
    round_transport = make_round([(gated, server)], lambda _: None, capacity=1, errors=errors)
    producer_waiting = threading.Event()

    def blocked_producer():
        producer_waiting.set()
        try:
            round_transport.reserve(0, lambda: None)
        except BaseException as error:
            producer_errors.append(error)

    producer = threading.Thread(target=blocked_producer)
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(packet())
        assert gated.entered.wait(1)
        producer.start()
        assert producer_waiting.wait(1)
        server.shutdown(socket.SHUT_RDWR)
        server.close()
        gated.release.set()
        wait_until(lambda: len(errors) == 1)
        producer.join(1)
        wait_until(lambda: round_transport.stats()['workers_alive'] == 0)
        assert len(producer_errors) == 1
        assert client.fileno() == server.fileno() == -1
        assert round_transport.stats()['pending_sd'] == round_transport.stats()['pending_gradients'] == 0
        with pytest.raises(RuntimeError):
            round_transport.finish()
        round_transport.abort(ValueError('second failure'))
        assert len(errors) == 1
    finally:
        gated.release.set()
        round_transport.abort()
        if producer.ident is not None:
            producer.join(1)


@pytest.mark.parametrize('alter', ['payload', 'job_id'])
def test_receiver_rejects_validly_framed_bytes_from_wrong_job_or_source(monkeypatch, alter):
    module = api()
    original_send = module.send_tensors

    def changed_send(sock, tensors, metadata):
        if metadata.get('kind') == 'sd':
            tensors, metadata = dict(tensors), dict(metadata)
            if alter == 'payload':
                tensors['activation'] = tensors['activation'] + 1
            else:
                metadata['job_id'] = [1, 0, 999]
        return original_send(sock, tensors, metadata)

    monkeypatch.setattr(module, 'send_tensors', changed_send)
    accepted, errors = [], []
    round_transport = make_round([socket.socketpair()], accepted.append, errors=errors)
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(packet())
        wait_until(lambda: errors)
        assert accepted == []
        with pytest.raises(RuntimeError):
            round_transport.finish()
        assert len(errors) == 1 and isinstance(errors[0], ValueError)
        assert round_transport.stats()['workers_alive'] == 0
    finally:
        round_transport.abort()


def test_job_id_cannot_be_resubmitted_after_a_upload_was_admitted():
    round_transport = make_round([socket.socketpair()], lambda _: None)
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(packet())
        wait_until(lambda: round_transport.stats()['pending_sd'] == 0)
        round_transport.reserve(0, lambda: None)
        with pytest.raises(ValueError):
            round_transport.submit(packet())
    finally:
        round_transport.abort()
        with pytest.raises(RuntimeError):
            round_transport.finish()


def test_default_three_credits_cover_queued_transmitting_and_unreceived_uploads():
    client, server = socket.socketpair()
    gated = PausedSocket(client, 'send')
    errors, submitted, producer_errors = [], [], []
    round_transport = make_round([(gated, server)], lambda _: None, errors=errors)
    fourth_compute = threading.Event()

    def producer():
        try:
            for index in range(4):
                round_transport.reserve(0, lambda: None)
                if index == 3:
                    fourth_compute.set()
                round_transport.submit(packet(index))
                submitted.append(index)
        except RuntimeError as error:
            producer_errors.append(error)

    thread = threading.Thread(target=producer)
    try:
        thread.start()
        assert gated.entered.wait(1)
        wait_until(lambda: len(submitted) == 3)
        assert submitted == [0, 1, 2]
        assert not fourth_compute.wait(.05)
        assert round_transport.stats()['pending_sd'] == 3
        round_transport.abort()
        thread.join(1)
        with pytest.raises(RuntimeError):
            round_transport.finish()
        assert len(producer_errors) == 1 and errors == []
        assert round_transport.stats()['max_pending_sd_per_client'] == [3]
        assert round_transport.stats()['workers_alive'] == 0
    finally:
        gated.release.set()
        round_transport.abort()
        thread.join(1)


def test_slow_client_upload_does_not_stall_another_client():
    client0, server0 = socket.socketpair()
    gated = PausedSocket(client0, 'send')
    accepted, errors = [], []
    round_transport = make_round([(gated, server0), socket.socketpair()], accepted.append, errors=errors)
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(packet())
        assert gated.entered.wait(1)
        item1 = packet()
        item1['row'].update(job_id=(1, 1, 0), client_id=1)
        round_transport.reserve(1, lambda: None)
        round_transport.submit(item1)
        wait_until(lambda: len(accepted) == 1)
        assert accepted[0]['row']['client_id'] == 1
        gated.release.set()
        round_transport.finish()
        assert len(accepted) == 2 and errors == []
        assert round_transport.stats()['max_pending_sd_per_client'] == [1, 1]
    finally:
        gated.release.set()
        round_transport.abort()


def test_socket_timeout_is_finite_and_reports_failure_once():
    client, server = socket.socketpair()
    gated = PausedSocket(client, 'send')
    accepted, errors = [], []
    round_transport = api().TCPRoundTransport([(gated, server)], accepted.append,
        lambda *_: None, errors.append, time.monotonic, worker_timeout_s=.1).start()
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(packet())
        assert gated.entered.wait(1)
        wait_until(lambda: errors)
        with pytest.raises(RuntimeError):
            round_transport.finish()
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
        assert accepted == []
        assert round_transport.stats()['workers_alive'] == 0
        assert client.fileno() == server.fileno() == -1
    finally:
        gated.release.set()
        round_transport.abort()


def test_admission_callback_failure_cancels_every_worker_and_releases_credit():
    errors = []

    def admission(_):
        raise ValueError('Server admission failed')

    connections = [socket.socketpair(), socket.socketpair()]
    round_transport = make_round(connections, admission, errors=errors)
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(packet())
        wait_until(lambda: errors)
        with pytest.raises(RuntimeError):
            round_transport.finish()
        assert len(errors) == 1 and str(errors[0]) == 'Server admission failed'
        assert round_transport.stats()['pending_sd'] == round_transport.stats()['workers_alive'] == 0
        assert all(sock.fileno() == -1 for pair in connections for sock in pair)
    finally:
        round_transport.abort()


@pytest.mark.parametrize('header', [
    b'{"version":1,"version":1,"metadata":{},"tensors":[],"payload_bytes":0}',
    b'{"version":1,"metadata":{"n":1e999},"tensors":[],"payload_bytes":0}',
    b'{"version":true,"metadata":{},"tensors":[],"payload_bytes":0}',
    b'{"version":1,"metadata":{},"tensors":[],"payload_bytes":true}',
])
def test_strict_json_rejects_ambiguous_or_nonfinite_headers(header):
    client, server = socket.socketpair()
    try:
        client.sendall(struct.pack('!Q', len(header)) + header)
        with pytest.raises(ValueError):
            api().recv_tensors(server)
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize('dtype,values', [
    (torch.bool, [True, False]), (torch.int8, [-1, 2]),
    (torch.int16, [-3, 4]), (torch.int32, [5, 6]),
    (torch.float16, [1., 2.]), (torch.bfloat16, [3., 4.]),
    (torch.float64, [5., 6.]),
])
def test_model_tensor_frame_supports_safe_buffer_dtypes(dtype, values):
    client, server = socket.socketpair()
    try:
        source = torch.tensor(values, dtype=dtype)
        api().send_tensors(client, {'state': source}, {})
        tensors, metadata = api().recv_tensors(server)
        assert metadata == {} and tensors['state'].dtype == dtype
        assert torch.equal(tensors['state'], source)
    finally:
        client.close()
        server.close()


def test_real_kernel_send_backpressure_does_not_block_client_local_work():
    client, server = socket.socketpair()
    client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2048)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
    first_received, admit = threading.Event(), threading.Event()
    accepted, errors = [], []

    def admission(item):
        if item['row']['job_id'][2] == 0:
            first_received.set()
            assert admit.wait(2)
        accepted.append(item)

    round_transport = make_round([(client, server)], admission, errors=errors)
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(packet())
        assert first_received.wait(1)
        round_transport.reserve(0, lambda: None)
        large = packet(1, size=8192)
        round_transport.submit(large)
        wait_until(lambda: 'sd_send_start_s' in large['row'])
        # The receiver cannot consume this frame until first SD admission. Its
        # 128KiB payload exceeds the deliberately small real kernel send buffer.
        round_transport.reserve(0, lambda: None)
        local_result = 21 * 2
        round_transport.submit(packet(2))
        assert local_result == 42
        assert 'sd_send_return_s' not in large['row']
        assert round_transport.stats()['pending_sd'] == 3
        admit.set()
        round_transport.finish()
        assert len(accepted) == 3 and errors == []
        assert round_transport.stats()['sd_payload_bytes_received'] == 131136
        assert round_transport.stats()['workers_alive'] == 0
    finally:
        admit.set()
        round_transport.abort()


def test_finish_timeout_aborts_pending_s_download_and_joins_workers_before_raising():
    accepted, errors = [], []
    client, server = socket.socketpair()
    round_transport = api().TCPRoundTransport([(client, server)], accepted.append,
        lambda *_: None, errors.append, time.monotonic, worker_timeout_s=.1).start()
    try:
        round_transport.reserve(0, lambda: None)
        round_transport.submit(packet(returns=True))
        wait_until(lambda: accepted)
        with pytest.raises(TimeoutError):
            round_transport.finish()
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
        assert round_transport.stats()['workers_alive'] == 0
        assert round_transport.stats()['pending_gradients'] == 0
        assert client.fileno() == server.fileno() == -1
    finally:
        round_transport.abort()

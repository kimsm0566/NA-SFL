"""Owned Mininet TCP links for NA-SFL, with no global network cleanup.

Only entering the context creates network resources. Mininet imports are lazy,
so ordinary CPU tests neither require Mininet nor modify the machine network.
CUDA models and workers remain in the caller's process: Linux sockets retain
their network namespace after the creating thread restores its original one.
"""
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import contextmanager
import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
import socket
import subprocess
import threading
import time
import uuid


_CLIENTS = 10
NETWORK_DEFAULTS = dict(slow_client_ids=[0, 5], fast_uplink_mbps=100,
                 fast_downlink_mbps=100, slow_uplink_mbps=20,
                 slow_downlink_mbps=50, rtt_ms=10, send_buffer_sd=3,
                 type='mininet', round_rate_variation=0., rate_seed=42)


def validate_network_config(config):
    if not isinstance(config, dict) or set(config) - set(NETWORK_DEFAULTS):
        raise ValueError('unsupported NA Mininet configuration')
    result = {**copy.deepcopy(NETWORK_DEFAULTS), **copy.deepcopy(config)}
    ids = result['slow_client_ids']
    if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < _CLIENTS for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('slow_client_ids must be distinct Client IDs in 0..9')
    for key in ('fast_uplink_mbps', 'fast_downlink_mbps', 'slow_uplink_mbps',
                'slow_downlink_mbps', 'rtt_ms'):
        value = result[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or (value < 0 if key == 'rtt_ms' else not 0 < value <= 1000)):
            raise ValueError('invalid ' + key)
    if type(result['send_buffer_sd']) is not int or result['send_buffer_sd'] < 1:
        raise ValueError('send_buffer_sd must be a positive integer')
    if result['type'] != 'mininet':
        raise ValueError('network type must be mininet')
    variation = result['round_rate_variation']
    if type(variation) not in (int, float) or not math.isfinite(variation) or not 0 <= variation < 1:
        raise ValueError('round_rate_variation must be finite and in [0,1)')
    if type(result['rate_seed']) is not int or not 0 <= result['rate_seed'] < 2**32:
        raise ValueError('rate_seed must be an integer in [0,2**32)')
    if any(result[key] * (1 + variation) > 1000 for key in (
            'fast_uplink_mbps', 'fast_downlink_mbps', 'slow_uplink_mbps', 'slow_downlink_mbps')):
        raise ValueError('variable TCLink access rate must stay <=1000Mbps')
    return result


def link_parameters(config, client_id):
    """TCLink params1 is Client egress UL; params2 is bridge egress DL."""
    config = validate_network_config(config)
    if type(client_id) is not int or not 0 <= client_id < _CLIENTS:
        raise ValueError('Client ID must be in 0..9')
    speed = 'slow' if client_id in config['slow_client_ids'] else 'fast'
    delay = '%gms' % (config['rtt_ms'] / 2)
    return tuple(dict(bw=config[speed + '_' + direction + '_mbps'],
                      delay=delay, max_queue_size=10000)
                 for direction in ('uplink', 'downlink'))


def round_link_rates(config, round_id):
    """Independent keyed draws; no training RNG, method, wall clock or call order."""
    config = validate_network_config(config)
    if type(round_id) is not int or round_id < 1:
        raise ValueError('network round must be a positive integer')
    clients = []
    for client_id in range(_CLIENTS):
        speed = 'slow' if client_id in config['slow_client_ids'] else 'fast'
        row = {'client_id': client_id}
        for direction in ('uplink', 'downlink'):
            key = f'na_sfl_round_rates_v1:{config["rate_seed"]}:{round_id}:{client_id}:{direction}'
            rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), 'big'))
            epsilon = rng.uniform(-config['round_rate_variation'], config['round_rate_variation'])
            row[direction + '_mbps'] = round(config[speed + '_' + direction + '_mbps'] * (1 + epsilon), 6)
        clients.append(row)
    return clients


def rate_scenario(config, rounds):
    """Full planned table, identical across methods and with stable round prefixes."""
    if type(rounds) is not int or rounds < 1:
        raise ValueError('network rounds must be a positive integer')
    config = validate_network_config(config)
    result = {'schema': 'na_sfl_round_rates_v1', 'config': config,
              'rate_precision_mbps': 1e-6,
              'rounds': [{'round': r, 'clients': round_link_rates(config, r)} for r in range(1, rounds + 1)]}
    encoded = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**result, 'sha256': hashlib.sha256(encoded).hexdigest()}


def _setns(fd):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.setns(ctypes.c_int(fd), ctypes.c_int(0x40000000)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


@contextmanager
def _network_namespace(path):
    """setns is thread-local; always restore this thread, including on failure."""
    original = os.open('/proc/thread-self/ns/net', os.O_RDONLY)
    target = None
    entered = False
    try:
        target = os.open(path, os.O_RDONLY)
        _setns(target)
        entered = True
        yield
    finally:
        try:
            if entered:
                _setns(original)
        finally:
            if target is not None:
                os.close(target)
            os.close(original)


def _run(*args):
    return subprocess.run(args, check=True, text=True, capture_output=True,
                          timeout=30).stdout


def _close_socket(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


class MininetNetwork:
    """Ten independent shaped access links and one unshaped Server link.

    Every bridge/interface name is unique and all topology construction takes
    place inside a newly owned outer namespace. No controller, OVS, physical
    interface, host routing, hwsim, or existing named namespace is modified.
    """
    def __init__(self, config: dict, result_dir: Path, rounds=None):
        self.config = validate_network_config(config)
        self.scenario = rate_scenario(self.config, rounds) if rounds is not None else None
        self.result_dir = Path(result_dir)
        suffix = uuid.uuid4().hex[:8]
        self.namespace = 'hsfl-na-' + suffix
        self._prefix = 'n' + suffix[:5]
        self._net = None
        self.clients = []
        self.server = None
        self.switch = None
        self._access_links = []
        self._server_link = None
        self._connections = []
        self._outer_created = False
        self._active = False
        self._closed = False
        self._lock = threading.Lock()
        self._metadata = dict(namespace=self.namespace, config=self.config,
                              transport='AF_INET/TCP', client_count=_CLIENTS,
                              server_common_link_rate_limit=None,
                              packet_loss_percent=0, jitter_ms=0,
                              active_round=None, round_rates=[],
                              rate_scenario_sha256=self.scenario['sha256'] if self.scenario else None,
                              model_traffic=[], tc_snapshots=[], cleanup_errors=[],
                              limitation='CPU/network readiness does not validate GPU training')

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError('network context cannot be entered again')
        try:
            self._start()
            self._active = True
            self.snapshot('started')
            return self
        except BaseException:
            self.close()
            raise

    def _start(self):
        from mininet.net import Mininet
        from mininet.nodelib import LinuxBridge
        from mininet.link import TCLink

        _run('ip', 'netns', 'add', self.namespace)
        self._outer_created = True
        with _network_namespace('/run/netns/' + self.namespace):
            self._net = Mininet(controller=None, switch=LinuxBridge, link=TCLink,
                                build=False, cleanup=False, ipBase='10.233.0.0/24',
                                autoStaticArp=True)
            self.switch = self._net.addSwitch(self._prefix + 'b',
                                              dpid='0000000000000001', stp=False)
            for client_id in range(_CLIENTS):
                client = self._net.addHost(self._prefix + 'c' + str(client_id),
                                           ip='10.233.0.%d/24' % (client_id + 1))
                self.clients.append(client)
                uplink, downlink = link_parameters(self.config, client_id)
                self._access_links.append(self._net.addLink(client, self.switch,
                                                            params1=uplink,
                                                            params2=downlink))
            self.server = self._net.addHost(self._prefix + 's', ip='10.233.0.254/24')
            # Omitting bw leaves the common Server path unshaped.
            self._server_link = self._net.addLink(self.server, self.switch)
            self._net.start()
        self._metadata['nodes'] = dict(
            clients=[dict(id=i, name=node.name, pid=node.pid, ip=node.IP())
                     for i, node in enumerate(self.clients)],
            server=dict(name=self.server.name, pid=self.server.pid, ip=self.server.IP()),
            bridge=dict(name=self.switch.name, pid=self.switch.pid))
        self._metadata['started_monotonic_s'] = time.monotonic()

    def _require_active(self):
        if not self._active or self._closed:
            raise RuntimeError('network must be active')

    def apply_round(self, round_id):
        """Change only owned HTB rate/ceil after previous TCP workers have drained."""
        self._require_active()
        clients = round_link_rates(self.config, round_id)
        if self.scenario is not None and round_id > len(self.scenario['rounds']):
            raise ValueError('network round exceeds planned scenario')
        if self._metadata['active_round'] == round_id:
            return copy.deepcopy(self._metadata['round_rates'][-1])
        if any(sock.fileno() >= 0 for pair in self._connections for sock in pair):
            raise RuntimeError('cannot change rates while TCP connections are open')
        started = time.monotonic()
        observed = []
        try:
            if self.config['round_rate_variation']:
                for row, link in zip(clients, self._access_links):
                    actual = {'client_id': row['client_id']}
                    for direction, intf in (('uplink', link.intf1), ('downlink', link.intf2)):
                        rate = row[direction + '_mbps']
                        text = f'{rate:.6f}Mbit'
                        _, error, code = intf.node.pexec('tc', 'class', 'change', 'dev', intf.name,
                            'parent', '5:0', 'classid', '5:1', 'htb', 'rate', text, 'ceil', text, 'burst', '15k')
                        if code:
                            raise RuntimeError('TC failure for ' + intf.name + ': ' + str(error))
                        classes = self._interface_snapshot(intf)['class']
                        tolerance = max(.001, rate * 1e-5)
                        if (len(classes) != 1 or classes[0]['kind'] != 'htb'
                                or any(abs(classes[0][key] - rate) > tolerance
                                       for key in ('rate_mbps', 'ceil_mbps'))):
                            raise RuntimeError('TC readback differs for ' + intf.name)
                        actual[direction + '_mbps'] = classes[0]['rate_mbps']
                    observed.append(actual)
        except Exception as error:
            self._metadata['rate_change_error'] = dict(round=round_id, error=repr(error))
            self._write_metadata()
            raise
        record = dict(round=round_id, clients=clients, kernel_rates=observed,
                      apply_wall_s=time.monotonic() - started,
                      rate_scenario_sha256=self._metadata['rate_scenario_sha256'])
        self._metadata['round_rates'].append(record)
        self._metadata['active_round'] = round_id
        self._write_metadata()
        return copy.deepcopy(record)

    @staticmethod
    def _socket_for(node):
        with _network_namespace('/proc/%d/ns/net' % node.pid):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(60)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    def create_connections(self):
        """Return (Client, Server) connected sockets for each fixed Client ID."""
        self._require_active()
        pairs = []
        listener = self._socket_for(self.server)
        pending = None
        try:
            listener.bind((self.server.IP(), 0))
            listener.listen(_CLIENTS)
            address = listener.getsockname()
            for client in self.clients:
                pending = self._socket_for(client)
                pending.bind((client.IP(), 0))
                pending.connect(address)
                with _network_namespace('/proc/%d/ns/net' % self.server.pid):
                    accepted, _ = listener.accept()
                accepted.settimeout(60)
                accepted.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                pairs.append((pending, accepted))
                pending = None
            self._connections.extend(pairs)
            return pairs
        except BaseException:
            if pending is not None:
                _close_socket(pending)
            for pair in pairs:
                for sock in pair:
                    _close_socket(sock)
            raise
        finally:
            _close_socket(listener)

    def transfer_models(self, states_by_client, direction, tag):
        """Send real CPU state_dict frames and return the decoded receiver states."""
        if direction not in ('upload', 'download'):
            raise ValueError('direction must be upload or download')
        if len(states_by_client) != _CLIENTS:
            raise ValueError('one state_dict is required for each of ten Clients')
        self._require_active()
        from utils.na_sfl_transport import send_tensors, recv_tensors

        pairs = self.create_connections()
        started = time.monotonic()
        decoded = [None] * _CLIENTS
        sent = [None] * _CLIENTS
        executor = ThreadPoolExecutor(max_workers=2 * _CLIENTS,
                                      thread_name_prefix='na-model-tcp')
        futures = []
        def receive(client_id, sock, expected):
            tensors, metadata = recv_tensors(sock)
            if metadata != expected:
                raise ValueError('model frame metadata mismatch')
            decoded[client_id] = tensors
        def send(client_id, sock, state, metadata):
            sent[client_id] = send_tensors(sock, state, metadata)
        try:
            for i, (client_sock, server_sock) in enumerate(pairs):
                source, destination = ((client_sock, server_sock) if direction == 'upload'
                                       else (server_sock, client_sock))
                metadata = dict(kind='model', direction=direction, tag=str(tag), client_id=i)
                futures.append(executor.submit(receive, i, destination, metadata))
                futures.append(executor.submit(send, i, source, states_by_client[i], metadata))
            completed, _ = wait(futures, return_when=FIRST_EXCEPTION)
            for future in completed:
                future.result()
        finally:
            for pair in pairs:
                for sock in pair:
                    _close_socket(sock)
                self._connections.remove(pair)
            executor.shutdown(wait=True, cancel_futures=True)
        record = dict(tag=str(tag), direction=direction,
                      wall_s=time.monotonic() - started, clients=sent,
                      payload_bytes=sum(item['payload_bytes'] for item in sent),
                      wire_bytes=sum(item['wire_bytes'] for item in sent))
        with self._lock:
            self._metadata['model_traffic'].append(record)
        self._write_metadata()
        return decoded

    @staticmethod
    def _interface_snapshot(intf):
        result = dict(node=intf.node.name, interface=intf.name)
        # iproute2 5.15 prints HTB classes as text even with -j. Keep its
        # unmodified counters and parse only the explicit rate/ceil fields.
        raw = intf.node.cmd('tc -s class show dev ' + intf.name)
        result['class_raw'] = raw
        result['class'] = []
        units = dict(bit=1e-6, Kbit=1e-3, Mbit=1., Gbit=1e3)
        for line in raw.splitlines():
            match = re.search(r'^class (\w+) .* rate ([\d.]+)([KMG]?bit) ceil ([\d.]+)([KMG]?bit)', line)
            if match:
                kind, rate, rate_unit, ceiling, ceiling_unit = match.groups()
                result['class'].append(dict(kind=kind,
                    rate_mbps=float(rate) * units[rate_unit],
                    ceil_mbps=float(ceiling) * units[ceiling_unit]))
        raw = intf.node.cmd('tc -j -s qdisc show dev ' + intf.name)
        result['qdisc'] = json.loads(raw or '[]')
        return result

    def snapshot(self, tag='snapshot'):
        """Record actual kernel TC rates and counters, separate from payload stats."""
        if self._active and self._net is not None:
            links = [dict(client_id=i,
                          uplink=self._interface_snapshot(link.intf1),
                          downlink=self._interface_snapshot(link.intf2))
                     for i, link in enumerate(self._access_links)]
            common = [self._interface_snapshot(intf)
                      for intf in (self._server_link.intf1, self._server_link.intf2)]
            self._metadata['tc_snapshots'].append(dict(tag=str(tag),
                monotonic_s=time.monotonic(), access_links=links, server_link=common))
        self._write_metadata()
        return self.stats()

    def stats(self):
        with self._lock:
            return copy.deepcopy({**self._metadata, 'active': self._active,
                                  'closed': self._closed})

    def _write_metadata(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        scenario = self.result_dir / 'network_rate_schedule.json'
        if self.scenario is not None and not scenario.exists():
            temporary = scenario.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(self.scenario, indent=2, sort_keys=True) + '\n')
            temporary.replace(scenario)
        target = self.result_dir / 'network_metadata.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.stats(), indent=2, sort_keys=True) + '\n')
        temporary.replace(target)

    def close(self):
        """Close owned sockets, Mininet nodes/links, and only our outer namespace."""
        if self._closed:
            return
        errors = self._metadata['cleanup_errors']
        for pair in self._connections:
            for sock in pair:
                _close_socket(sock)
        self._connections.clear()
        if self._active:
            try:
                self.snapshot('before_close')
            except Exception as error:
                errors.append('snapshot: ' + repr(error))
        if self._net is not None:
            owned_processes = [(node, getattr(node, 'shell', None))
                               for node in self._net.hosts + self._net.switches]
            try:
                with _network_namespace('/run/netns/' + self.namespace):
                    self._net.stop()
            except Exception as error:
                errors.append('Mininet stop: ' + repr(error))
            # These are only Popen handles owned by this topology, never a name scan.
            for node, process in owned_processes:
                if process is not None:
                    try:
                        if process.poll() is None and getattr(node, 'shell', None) is not None:
                            node.terminate()
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    except Exception as error:
                        errors.append('node ' + node.name + ': ' + repr(error))
        if self._outer_created:
            try:
                _run('ip', 'netns', 'delete', self.namespace)
                self._outer_created = False
            except Exception as error:
                errors.append('namespace: ' + repr(error))
        self._active = False
        self._closed = True
        self._metadata['closed_monotonic_s'] = time.monotonic()
        self._write_metadata()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is None and self._metadata['cleanup_errors']:
            raise RuntimeError('network cleanup had errors: ' + repr(self._metadata['cleanup_errors']))
        return False

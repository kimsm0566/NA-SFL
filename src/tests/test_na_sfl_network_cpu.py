"""Network contracts without creating a privileged topology during pytest."""
import socket
from pathlib import Path

import pytest

from utils import na_sfl_network as network


CONFIG = dict(slow_client_ids=[0, 5], fast_uplink_mbps=100,
              fast_downlink_mbps=100, slow_uplink_mbps=20,
              slow_downlink_mbps=50, rtt_ms=10, send_buffer_sd=3)


def test_access_link_applies_ul_to_client_dl_to_switch_and_half_rtt():
    for client_id in range(10):
        uplink, downlink = network.link_parameters(CONFIG, client_id)
        assert uplink == dict(bw=20 if client_id in (0, 5) else 100,
                              delay='5ms', max_queue_size=10000)
        assert downlink == dict(bw=50 if client_id in (0, 5) else 100,
                                delay='5ms', max_queue_size=10000)


@pytest.mark.parametrize('patch', [dict(slow_client_ids=[-1]),
    dict(slow_client_ids=[True]), dict(slow_client_ids=[0, 0]),
    dict(fast_uplink_mbps=0), dict(slow_downlink_mbps=float('nan')),
    dict(rtt_ms=-1), dict(send_buffer_sd=True), dict(type='wifi')])
def test_invalid_configuration_fails_before_network_mutation(tmp_path, patch):
    with pytest.raises(ValueError):
        network.MininetNetwork({**CONFIG, **patch}, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_namespace_restores_calling_thread_on_body_exception(monkeypatch):
    opened, changed, closed = [], [], []
    def fake_open(path, flags):
        opened.append(str(path))
        return len(opened) + 10
    monkeypatch.setattr(network.os, 'open', fake_open)
    monkeypatch.setattr(network.os, 'close', closed.append)
    monkeypatch.setattr(network, '_setns', changed.append)
    with pytest.raises(RuntimeError, match='body failed'):
        with network._network_namespace('/run/netns/owned'):
            raise RuntimeError('body failed')
    assert opened == ['/proc/thread-self/ns/net', '/run/netns/owned']
    assert changed == [12, 11]
    assert sorted(closed) == [11, 12]


def test_namespace_closes_original_fd_when_target_open_fails(monkeypatch):
    closed = []
    def fake_open(path, flags):
        if str(path) == '/run/netns/missing':
            raise FileNotFoundError(path)
        return 19
    monkeypatch.setattr(network.os, 'open', fake_open)
    monkeypatch.setattr(network.os, 'close', closed.append)
    with pytest.raises(FileNotFoundError):
        with network._network_namespace('/run/netns/missing'):
            pass
    assert closed == [19]


def test_enter_failure_closes_only_owned_resources_and_keeps_error(tmp_path, monkeypatch):
    topology = network.MininetNetwork(CONFIG, tmp_path)
    first, second = socket.socketpair()
    topology._connections.append((first, second))
    monkeypatch.setattr(topology, '_start', lambda: (_ for _ in ()).throw(RuntimeError('start failed')))
    with pytest.raises(RuntimeError, match='start failed'):
        topology.__enter__()
    assert first.fileno() == second.fileno() == -1
    assert topology.stats()['closed'] is True
    topology.close()


def test_unstarted_topology_cannot_create_connections_or_transfer(tmp_path):
    topology = network.MininetNetwork(CONFIG, tmp_path)
    with pytest.raises(RuntimeError, match='active'):
        topology.create_connections()
    with pytest.raises(ValueError):
        topology.transfer_models([], 'wrong', 'test')
    with pytest.raises(ValueError):
        topology.transfer_models([], 'upload', 'test')
    topology.close()
    assert topology.namespace.startswith('hsfl-na-')
    assert (tmp_path / 'network_metadata.json').is_file()


def test_tc_snapshot_accepts_installed_htb_text_classes_and_json_qdisc():
    from types import SimpleNamespace
    class_text = ('class htb 5:1 root leaf 10: prio 0 rate 20Mbit ceil 20Mbit burst 15Kb\n'
                  ' Sent 123 bytes 2 pkt (dropped 0, overlimits 1 requeues 0)\n')
    def command(text):
        return class_text if 'class' in text else '[{"kind":"netem","bytes":123}]'
    intf = SimpleNamespace(name='own-eth0', node=SimpleNamespace(name='own', cmd=command))
    snapshot = network.MininetNetwork._interface_snapshot(intf)
    assert snapshot['class'] == [dict(kind='htb', rate_mbps=20., ceil_mbps=20.)]
    assert snapshot['class_raw'] == class_text
    assert snapshot['qdisc'] == [dict(kind='netem', bytes=123)]


def test_close_reaps_owned_shell_even_after_mininet_clears_node_shell(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    waits = []
    process = SimpleNamespace(poll=lambda: 0, wait=lambda timeout: waits.append(timeout))
    node = SimpleNamespace(name='owned', shell=process)
    topology = network.MininetNetwork(CONFIG, tmp_path)
    topology._net = SimpleNamespace(hosts=[node], switches=[],
                                    stop=lambda: setattr(node, 'shell', None))
    monkeypatch.setattr(network, '_network_namespace', lambda path: nullcontext())
    topology.close()
    assert waits == [5]
    assert topology.stats()['cleanup_errors'] == []


def test_model_send_error_aborts_receivers_and_closes_all_temporary_sockets(tmp_path, monkeypatch):
    from utils import na_sfl_transport
    topology = network.MininetNetwork(CONFIG, tmp_path)
    topology._active = True
    pairs = [socket.socketpair() for _ in range(10)]
    for pair in pairs:
        for sock in pair:
            sock.settimeout(1)
    def connections():
        topology._connections.extend(pairs)
        return pairs
    def fail_send(*args):
        raise ValueError('injected send failure')
    monkeypatch.setattr(topology, 'create_connections', connections)
    monkeypatch.setattr(na_sfl_transport, 'send_tensors', fail_send)
    with pytest.raises(ValueError, match='injected send failure'):
        topology.transfer_models([{}] * 10, 'upload', 'failure')
    assert topology._connections == []
    assert all(sock.fileno() == -1 for pair in pairs for sock in pair)
    assert topology.stats()['model_traffic'] == []
    topology.close()


def test_mid_connection_failure_closes_pending_listener_and_previous_pairs(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    class OwnedSocket:
        def __init__(self, fail=False):
            self.closed, self.fail = False, fail
        def bind(self, address): pass
        def listen(self, count): pass
        def getsockname(self): return ('10.233.0.254', 12345)
        def connect(self, address):
            if self.fail: raise OSError('injected connect failure')
        def accept(self): return accepted, ('10.233.0.1', 12346)
        def settimeout(self, timeout): pass
        def setsockopt(self, *args): pass
        def shutdown(self, how): pass
        def close(self): self.closed = True
    listener, first, second, accepted = (OwnedSocket(), OwnedSocket(),
                                         OwnedSocket(fail=True), OwnedSocket())
    topology = network.MininetNetwork(CONFIG, tmp_path)
    topology._active = True
    topology.server = SimpleNamespace(pid=1, IP=lambda: '10.233.0.254')
    topology.clients = [SimpleNamespace(IP=lambda: '10.233.0.1'),
                        SimpleNamespace(IP=lambda: '10.233.0.2')]
    sockets = iter((listener, first, second))
    monkeypatch.setattr(topology, '_socket_for', lambda node: next(sockets))
    monkeypatch.setattr(network, '_network_namespace', lambda path: nullcontext())
    with pytest.raises(OSError, match='injected connect failure'):
        topology.create_connections()
    assert all(sock.closed for sock in (listener, first, second, accepted))
    assert topology._connections == []
    topology.close()


def test_round_rates_are_bounded_independent_and_do_not_consume_training_rng():
    import random
    config = {**CONFIG, 'round_rate_variation': .1, 'rate_seed': 42}
    before = random.getstate()
    rounds = [network.round_link_rates(config, r) for r in range(1, 101)]
    assert random.getstate() == before
    for clients in rounds:
        assert [row['client_id'] for row in clients] == list(range(10))
        for row in clients:
            slow = row['client_id'] in (0, 5)
            assert (18 if slow else 90) <= row['uplink_mbps'] <= (22 if slow else 110)
            assert (45 if slow else 90) <= row['downlink_mbps'] <= (55 if slow else 110)
    fast = [row for row in rounds[9] if row['client_id'] not in (0, 5)]
    assert len({row['uplink_mbps'] for row in fast}) == 8
    assert all(row['uplink_mbps'] != row['downlink_mbps'] for row in fast)
    assert rounds[0] != rounds[1]
    assert network.round_link_rates(config, 10) == rounds[9]
    assert network.round_link_rates({**config, 'rate_seed': 43}, 10) != rounds[9]


def test_zero_variation_keeps_existing_fixed_rates_for_every_round():
    for r in (1, 10, 100):
        clients = network.round_link_rates(CONFIG, r)
        assert [(row['uplink_mbps'], row['downlink_mbps']) for row in clients] == [
            (20, 50), (100, 100), (100, 100), (100, 100), (100, 100),
            (20, 50), (100, 100), (100, 100), (100, 100), (100, 100)]


def test_four_methods_share_schedule_and_resume_or_longer_budget_keeps_prefix():
    from utils.na_sfl_config import load_config, comparison_configs
    cfg = load_config(overrides={'network': {**CONFIG, 'round_rate_variation': .1, 'rate_seed': 42}})
    scenarios = [network.rate_scenario(c['network'], 100) for c in comparison_configs(cfg)]
    assert all(s == scenarios[0] for s in scenarios)
    assert len(scenarios[0]['rounds']) == 100
    assert len(scenarios[0]['sha256']) == 64
    assert network.rate_scenario(cfg['network'], 10)['rounds'] == scenarios[0]['rounds'][:10]
    # Jumping directly to a resumed round must not replay earlier RNG calls.
    assert network.round_link_rates(cfg['network'], 10) == scenarios[0]['rounds'][9]['clients']
    assert network.rate_scenario({**cfg['network'], 'rate_seed': 43}, 100)['sha256'] != scenarios[0]['sha256']


@pytest.mark.parametrize('patch', [
    {'round_rate_variation': -0.1}, {'round_rate_variation': 1.0},
    {'round_rate_variation': float('nan')}, {'round_rate_variation': True},
    {'rate_seed': -1}, {'rate_seed': True}, {'rate_seed': 2**32},
    {'rate_seed': 1.5}, {'round_rate_variation': .1, 'fast_uplink_mbps': 950}])
def test_invalid_rate_scenario_is_rejected_by_config_and_topology(tmp_path, patch):
    from utils.na_sfl_config import load_config
    config = {**CONFIG, 'round_rate_variation': .1, 'rate_seed': 42, **patch}
    with pytest.raises(ValueError):
        load_config(overrides={'network': config})
    with pytest.raises(ValueError):
        network.MininetNetwork(config, tmp_path, rounds=100)
    assert not list(tmp_path.iterdir())


def test_rate_change_refuses_live_sockets_and_tc_failure_never_commits_round(tmp_path):
    from types import SimpleNamespace
    config = {**CONFIG, 'round_rate_variation': .1, 'rate_seed': 42}
    topology = network.MininetNetwork(config, tmp_path, rounds=100)
    topology._active = True
    first, second = socket.socketpair()
    topology._connections.append((first, second))
    try:
        with pytest.raises(RuntimeError, match='connections'):
            topology.apply_round(1)
    finally:
        first.close()
        second.close()
    failed = SimpleNamespace(name='owned-eth0', node=SimpleNamespace(
        pexec=lambda *args: ('', 'injected TC failure', 2)))
    topology._access_links = [SimpleNamespace(intf1=failed, intf2=failed)] * 10
    with pytest.raises(RuntimeError, match='TC failure'):
        topology.apply_round(1)
    assert topology.stats()['active_round'] is None
    assert topology.stats()['round_rates'] == []


@pytest.mark.parametrize('completed', [0, 9])
def test_coordinator_applies_rates_before_initial_transfer_and_each_round(tmp_path, monkeypatch, completed):
    """Exercise real round orchestration; stub model/CUDA work only."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    from utils import na_sfl_train as train
    events = []
    class LinkObserver:
        current = None
        def apply_round(self, number):
            self.current = number
            events.append(('rates', number))
            return {'round': number, 'clients': []}
        def transfer_models(self, packets, direction, tag):
            events.append(('initial_transfer', self.current))
            return packets
    owner = object.__new__(train.NASFLTrainer)
    owner.args = dict(rounds=completed + 2, server_max_batch_samples=512,
        server_policy='individual', method='splitfedv2', seed=42,
        aux_client_count=0, aux_client_ids=[])
    owner.network = LinkObserver()
    owner.sources, owner.data_sha256, owner.partitions = {}, None, None
    owner.next_round = completed
    owner.rounds = [{'elapsed_wall_s': 0.}] if completed else []
    owner.initial_evaluation = {'test_accuracy': 10.} if completed else None
    owner.clients = [SimpleNamespace(model=SimpleNamespace(train=lambda: None), auxiliary_head=None)]
    owner.server = SimpleNamespace(model=SimpleNamespace(train=lambda: None), version=0)
    owner.device, owner.run_dir = 'cpu', tmp_path
    owner.server_batch_records, owner.execution_boundaries = [], []
    owner._model_packets = lambda: [{}]
    owner._load_model_packets = lambda packets: None
    owner._evaluate = lambda *args: (10., 2., None)
    owner._validate_jobs = lambda *args: 1
    owner._aggregate = lambda weights, r: events.append(('fedavg', r, owner.network.current))
    owner._progress_files = lambda: None
    owner._log = lambda message: None
    owner.checkpoint_payload = lambda: {}
    def one_round(instance, images, labels, parts, number):
        events.append(('train', number, instance.network.current))
        instance.server.version += 1
        instance.last_async_batches, instance.last_async_boundary = [{'samples': 1}], {}
        return [dict(client_id=0, samples=1, gradient_mode='server', aux_loss=None,
                     auxiliary_updated=False, loss=2.)]
    monkeypatch.setattr(train, 'run_async_round', one_round)
    monkeypatch.setattr(train, 'validate_server_batches', lambda *args: 1)
    monkeypatch.setattr(train, 'check_sources_unchanged', lambda sources: None)
    monkeypatch.setattr(train, 'sflv2_fp32', nullcontext)
    monkeypatch.setattr(train, 'atomic_checkpoint', lambda *args: None)
    for name in ('reset_peak_memory_stats', 'max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(train.torch.cuda, name, lambda *args: 0)
    images, labels = np.zeros((1, 1)), np.zeros(1, dtype=int)
    owner.run(images, labels, images, labels, [np.array([0])])
    assert events[0] == ('rates', completed + 1)
    assert events[1] == ('initial_transfer', completed + 1)
    for r in (completed + 1, completed + 2):
        assert ('train', r, r) in events
        assert ('fedavg', r, r) in events
    assert not train.torch.cuda.is_initialized()

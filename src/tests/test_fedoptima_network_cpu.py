"""Streaming rate transitions using fake HTB interfaces; no real network/GPU."""
import importlib
import json
import threading
from types import SimpleNamespace

import pytest

from utils import na_sfl_network as base


CONFIG = {**base.NETWORK_DEFAULTS, "round_rate_variation": .1, "rate_seed": 42}


@pytest.fixture
def adapter():
    # Resolve lazily so the initial RED names the missing adapter explicitly.
    try:
        return importlib.import_module("reference_external.FedOptima.hsfl.network")
    except ModuleNotFoundError:
        pytest.fail("FedOptima streaming network adapter is not implemented")


class FakeSocket:
    def __init__(self):
        self.closed = False

    def fileno(self):
        return -1 if self.closed else 37

    def shutdown(self, how):
        pass

    def close(self):
        self.closed = True


class FakeNode:
    def __init__(self, name):
        self.name = name
        self.commands = []
        self.rate = 0.
        self.ceil = 0.
        self.error = False
        self.readback = "correct"

    def pexec(self, *args):
        self.commands.append(args)
        if self.error:
            return "", "injected TC failure", 2
        self.rate = float(args[args.index("rate") + 1][:-4])
        self.ceil = float(args[args.index("ceil") + 1][:-4])
        return "", "", 0

    def cmd(self, command):
        if "qdisc" in command:
            return "[]"
        if self.readback == "missing":
            return ""
        rate = self.rate + (1 if self.readback == "wrong_rate" else 0)
        ceil = self.ceil + (1 if self.readback == "wrong_ceil" else 0)
        if self.readback == "nan":
            return "class htb 5:1 root rate 20Mbit ceil 20Mbit\n"
        return f"class htb 5:1 root rate {rate:.6f}Mbit ceil {ceil:.6f}Mbit\n"


def prepared(adapter, tmp_path, monkeypatch):
    network = adapter.StreamingNetwork(CONFIG, tmp_path)
    nodes = []
    links = []
    for client in range(10):
        uplink = FakeNode(f"owned-client-{client}")
        downlink = FakeNode(f"owned-bridge-{client}")
        nodes.extend([uplink, downlink])
        links.append(SimpleNamespace(
            intf1=SimpleNamespace(name=f"c{client}-eth0", node=uplink),
            intf2=SimpleNamespace(name=f"b-eth{client}", node=downlink)))
    def start():
        network._access_links = links
    monkeypatch.setattr(network, "_start", start)
    return network, nodes


def test_constructor_has_no_namespace_or_file_side_effects(adapter, tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_run", lambda *a: pytest.fail("constructor ran network command"))
    network = adapter.StreamingNetwork(CONFIG, tmp_path)
    assert network.scenario is None
    assert not list(tmp_path.iterdir())
    assert network.stats()["rate_change_mode"] == "streaming_open_tcp"
    assert "drain" in network.stats()["rate_change_limitation"]


def test_round_rates_audit_all_twenty_and_preserve_open_sockets(adapter, tmp_path, monkeypatch):
    network, nodes = prepared(adapter, tmp_path, monkeypatch)
    sockets = [(FakeSocket(), FakeSocket()) for _ in range(10)]
    with network:
        network._connections.extend(sockets)
        for round_id in (1, 2):
            row = network.apply_live_round(round_id)
            assert row["clients"] == base.round_link_rates(CONFIG, round_id)
            assert len(row["kernel_rates"]) == 10
            for expected, observed in zip(row["clients"], row["kernel_rates"]):
                for direction in ("uplink", "downlink"):
                    assert observed[direction + "_mbps"] == expected[direction + "_mbps"]
                    assert observed[direction + "_ceil_mbps"] == expected[direction + "_mbps"]
            assert all(sock.fileno() >= 0 for pair in sockets for sock in pair)
        assert network._connections == sockets
        assert sum(len(node.commands) for node in nodes) == 40
        stored = json.loads((tmp_path / "network_metadata.json").read_text())
        assert stored["active_round"] == 2
        assert len(stored["round_rates"]) == 2
        for node in nodes:
            for cmd in node.commands:
                assert cmd[:4] == ("tc", "class", "change", "dev")
                assert cmd[6:10] == ("5:0", "classid", "5:1", "htb")
    assert all(sock.closed for pair in sockets for sock in pair)


def test_rounds_ordered_duplicate_idempotent_and_not_capped_at100(adapter, tmp_path, monkeypatch):
    network, nodes = prepared(adapter, tmp_path, monkeypatch)
    with network:
        for round_id in range(1, 102):
            network.apply_live_round(round_id)
        count = sum(len(node.commands) for node in nodes)
        repeated = network.apply_live_round(101)
        repeated["clients"].clear()
        assert len(network.stats()["round_rates"][-1]["clients"]) == 10
        assert sum(len(node.commands) for node in nodes) == count
        for invalid in (100, 103, 0, -1, True, 1.2):
            with pytest.raises(ValueError):
                network.apply_live_round(invalid)
        assert sum(len(node.commands) for node in nodes) == count
        assert len(network.stats()["round_rates"]) == 101


@pytest.mark.parametrize("readback", ["wrong_rate", "wrong_ceil", "missing"])
def test_wrong_or_partial_readback_aborts_without_committing_round(adapter, tmp_path, monkeypatch, readback):
    network, nodes = prepared(adapter, tmp_path, monkeypatch)
    nodes[-1].readback = readback
    with network:
        with pytest.raises(RuntimeError, match="readback"):
            network.apply_live_round(1)
        stats = network.stats()
        assert stats["active_round"] is None
        assert stats["round_rates"] == []
        assert stats["rate_change_error"]["round"] == 1
        assert len(stats["rate_change_error"]["completed_directions"]) == 19
        with pytest.raises(RuntimeError, match="failed"):
            network.apply_live_round(1)


def test_tc_failure_propagates_and_context_closes_owned_sockets(adapter, tmp_path, monkeypatch):
    network, nodes = prepared(adapter, tmp_path, monkeypatch)
    sockets = [(FakeSocket(), FakeSocket())]
    nodes[7].error = True
    with pytest.raises(RuntimeError, match="injected TC failure"):
        with network:
            network._connections.extend(sockets)
            network.apply_live_round(1)
    assert all(sock.closed for pair in sockets for sock in pair)
    stats = network.stats()
    assert stats["closed"] and not stats["active"]
    assert stats["round_rates"] == []
    assert len(stats["rate_change_error"]["completed_directions"]) == 7
    assert stats["cleanup_errors"] == []


def test_missing_access_link_aborts_before_any_tc_command(adapter, tmp_path, monkeypatch):
    network, nodes = prepared(adapter, tmp_path, monkeypatch)
    with network:
        network._access_links.pop()
        with pytest.raises(RuntimeError, match="ten"):
            network.apply_live_round(1)
        assert not any(node.commands for node in nodes)


def test_metadata_stats_and_writes_can_run_during_rate_changes(adapter, tmp_path, monkeypatch):
    network, nodes = prepared(adapter, tmp_path, monkeypatch)
    errors = []
    with network:
        def update():
            try:
                for round_id in range(1, 11):
                    network.apply_live_round(round_id)
            except BaseException as error:
                errors.append(error)
        def statistics():
            try:
                for _ in range(20):
                    network.stats()
                    network.snapshot("concurrent")
            except BaseException as error:
                errors.append(error)
        threads = [threading.Thread(target=fn, daemon=True) for fn in (update, statistics)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert not any(thread.is_alive() for thread in threads), "metadata lock deadlocked"
        assert not errors
        stored = json.loads((tmp_path / "network_metadata.json").read_text())
        assert stored["active_round"] == 10
        assert len(stored["round_rates"]) == 10


def test_inherited_connections_create_all_sockets_in_node_namespaces(adapter, tmp_path, monkeypatch):
    from contextlib import contextmanager

    network, nodes = prepared(adapter, tmp_path, monkeypatch)
    namespace = []
    created = []

    @contextmanager
    def enter(path):
        namespace.append(path)
        try:
            yield
        finally:
            namespace.pop()

    class NamespaceSocket(FakeSocket):
        def __init__(self, *args):
            super().__init__()
            assert namespace, "socket was created in caller/host namespace"
            self.created_in = namespace[-1]
            created.append(self)

        def settimeout(self, value):
            pass

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            pass

        def listen(self, count):
            pass

        def getsockname(self):
            return ("10.233.0.254", 10000)

        def connect(self, address):
            pass

        def accept(self):
            return NamespaceSocket(), ("10.233.0.1", 10001)

    monkeypatch.setattr(base, "_network_namespace", enter)
    monkeypatch.setattr(base.socket, "socket", NamespaceSocket)
    with network:
        network.clients = [SimpleNamespace(pid=100 + i, IP=lambda: "10.233.0.1")
                           for i in range(10)]
        network.server = SimpleNamespace(pid=200, IP=lambda: "10.233.0.254")
        pairs = network.create_connections()
        assert len(pairs) == 10
        for client_id, (client, server) in enumerate(pairs):
            assert client.created_in == f"/proc/{100 + client_id}/ns/net"
            assert server.created_in == "/proc/200/ns/net"
        assert len(created) == 21
        assert created[0].closed  # listener
        assert not namespace
        assert not any(sock.closed for pair in pairs for sock in pair)
        network.apply_live_round(1)
        assert not any(sock.closed for pair in pairs for sock in pair)
    assert all(sock.closed for sock in created)


def test_enter_failure_uses_inherited_owned_namespace_cleanup(adapter, tmp_path, monkeypatch):
    network = adapter.StreamingNetwork(CONFIG, tmp_path)
    deleted = []
    owned = [(FakeSocket(), FakeSocket())]

    def start_failure():
        network._outer_created = True
        network._connections.extend(owned)
        raise RuntimeError("topology startup failed")

    monkeypatch.setattr(network, "_start", start_failure)
    monkeypatch.setattr(base, "_run", lambda *args: deleted.append(args) or "")
    with pytest.raises(RuntimeError, match="topology startup failed"):
        with network:
            pytest.fail("failed context entered")
    assert deleted == [("ip", "netns", "delete", network.namespace)]
    assert all(sock.closed for pair in owned for sock in pair)
    assert network.stats()["closed"]
    assert network.stats()["cleanup_errors"] == []
    network.close()
    assert len(deleted) == 1


def test_requires_active_context_and_first_round_one(adapter, tmp_path, monkeypatch):
    network, nodes = prepared(adapter, tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="active"):
        network.apply_live_round(1)
    with network:
        with pytest.raises(ValueError, match="consecutive"):
            network.apply_live_round(2)
        assert not any(node.commands for node in nodes)
        network.apply_live_round(1)
    with pytest.raises(RuntimeError, match="active"):
        network.apply_live_round(2)

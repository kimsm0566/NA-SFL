"""FedOptima streaming use of the existing owned Mininet TCP topology.

The inherited connection factory creates every socket inside its Client/Server
network namespace. Rate transitions never replace those sockets or drain jobs.
No Mininet imports or namespace commands run until entering the context.
"""
import copy
import math
import threading
import time

from utils.na_sfl_network import MininetNetwork, round_link_rates


class StreamingNetwork(MininetNetwork):
    """Apply keyed per-round HTB rates while asynchronous TCP jobs continue.

    Call ``apply_live_round(1)`` before the first model distribution, then call
    once per global 30,000-visit boundary. The most recent round is idempotent;
    skipped or backwards transitions fail. There is no planned round ceiling.
    A failed transition leaves potentially mixed kernel rates, records partial
    progress and prohibits further changes; the caller must abort and close.
    """

    def __init__(self, config, result_dir):
        super().__init__(config, result_dir, rounds=None)
        # Base _write_metadata calls stats(), which also acquires _lock.
        # Reentrancy allows one lock to protect both metadata mutation and the
        # complete atomic-file write, including concurrent stats/snapshot calls.
        self._lock = threading.RLock()
        self._rate_change_failed = False
        self._metadata.update(
            rate_change_mode="streaming_open_tcp",
            rate_change_limitation=(
                "FedOptima changes owned HTB rate/ceil with existing TCP jobs "
                "and sockets open; NA changes after a drained round boundary. "
                "The 20 directions change sequentially, without a client barrier; "
                "in-flight packets can cross the rate boundary."),
            round_index_basis="global_completed_sample_visits_30000",
            rate_scenario_kind="unbounded_keyed_rounds_recorded_on_application",
        )

    def apply_live_round(self, round_id):
        """Update and audit all twenty owned access rates; preserve open TCP."""
        with self._lock:
            self._require_active()
            clients = round_link_rates(self.config, round_id)
            if self._rate_change_failed:
                raise RuntimeError("previous streaming rate transition failed; close network")
            current = self._metadata["active_round"]
            if current == round_id:
                return copy.deepcopy(self._metadata["round_rates"][-1])
            if round_id != (1 if current is None else current + 1):
                raise ValueError("streaming network rounds must be consecutive from one")
            if len(self._access_links) != 10:
                raise RuntimeError("streaming rate transition requires ten owned access links")
            started = time.monotonic()
            observed = []
            completed = []
            try:
                for row, link in zip(clients, self._access_links):
                    actual = {"client_id": row["client_id"]}
                    for direction, intf in (("uplink", link.intf1), ("downlink", link.intf2)):
                        rate = row[direction + "_mbps"]
                        text = f"{rate:.6f}Mbit"
                        _, error, code = intf.node.pexec(
                            "tc", "class", "change", "dev", intf.name,
                            "parent", "5:0", "classid", "5:1", "htb",
                            "rate", text, "ceil", text, "burst", "15k")
                        if code:
                            raise RuntimeError("TC failure for " + intf.name + ": " + str(error))
                        snapshot = self._interface_snapshot(intf)
                        classes = snapshot["class"]
                        tolerance = max(.001, rate * 1e-5)
                        if (len(classes) != 1 or classes[0].get("kind") != "htb"
                                or any(not isinstance(classes[0].get(key), (int, float))
                                       or not math.isfinite(classes[0][key])
                                       or abs(classes[0][key] - rate) > tolerance
                                       for key in ("rate_mbps", "ceil_mbps"))):
                            raise RuntimeError("TC readback differs for " + intf.name)
                        actual[direction + "_mbps"] = classes[0]["rate_mbps"]
                        actual[direction + "_ceil_mbps"] = classes[0]["ceil_mbps"]
                        completed.append(dict(client_id=row["client_id"], direction=direction,
                                              **snapshot))
                    observed.append(actual)
            except BaseException as error:
                self._rate_change_failed = True
                self._metadata["rate_change_error"] = dict(
                    round=round_id, error=repr(error),
                    completed_directions=completed,
                    kernel_rates_may_be_partial=True,
                    apply_wall_s=time.monotonic() - started)
                self._write_metadata()
                raise
            record = dict(round=round_id, clients=clients, kernel_rates=observed,
                          kernel_audit=completed,
                          apply_wall_s=time.monotonic() - started,
                          started_monotonic_s=started,
                          rate_change_mode="streaming_open_tcp",
                          rate_scenario_sha256=None)
            self._metadata["round_rates"].append(record)
            self._metadata["active_round"] = round_id
            self._write_metadata()
            return copy.deepcopy(record)

    def _write_metadata(self):
        with self._lock:
            return super()._write_metadata()

    def snapshot(self, tag="snapshot"):
        with self._lock:
            return super().snapshot(tag)

    def create_connections(self):
        with self._lock:
            return super().create_connections()

    def close(self):
        with self._lock:
            return super().close()

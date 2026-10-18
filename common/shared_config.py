"""Jetson-coordinated shared overlays. No hardware or mode-task transport.

The link carries ConfigCommand/ConfigReply objects in both directions over a
PC-initiated DEALER connection. Each socket belongs to one dedicated I/O thread;
validation, filesystem work and bounded RPC waits run elsewhere.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import logging
import queue
import threading
import time
import uuid

import zmq

from common.config import ConfigError
from common.config_schema import FIELDS
from common.live_config import (atomic_write, digest, flatten, put, read_overlay,
                                remove, unit_stacks, uses_key, writer_lock)

LOG = logging.getLogger(__name__)
RPC_TIMEOUT_S = 2.0


def shared_keys(root):
    """Ownership is derived from deployed stacks, including inactive units."""
    stacks = unit_stacks(root)
    return frozenset(k for k, spec in FIELDS.items() if "map" not in spec["types"]
                     and "[]" not in k and spec["tier"] != "fixed"
                     and len({u.split("/")[0] for u, stack in stacks.items()
                              if uses_key(stack["module"], k)}) > 1)


class SharedConfigAgent:
    """Serialize transactions and retain durable prepare/decision evidence.

    A lost commit reply is an uncertain decision, never a validation rejection.
    Reconnection repeats the durable commit before reconciling newer revisions.
    """
    def __init__(self, local):
        self.local = local
        self.keys = shared_keys(local.root)
        self.lock = threading.RLock()
        self.peer = None
        self.state_path = local.path.with_name("config-shared.json")
        try:
            self.state = json.loads(self.state_path.read_text())
        except FileNotFoundError:
            self.state = dict(ready=False, shared={}, hosts={}, prepared=None, decision=None)
        if local.host == "jetson" and self.state.get("decision"):
            decision = self.state["decision"]
            plan = decision["plan"]
            actual = digest(read_overlay(local.path))
            if actual == plan["old_digest"]:
                with writer_lock(local.path):
                    local.apply_plan(plan)
            elif actual != plan["resulting_digest"]:
                raise ConfigError("overlay differs from durable shared decision; manual recovery required")
        self.state.setdefault("revision_transaction", uuid.uuid4().hex)
        if local.host == "jetson" and self.state.get("decision"):
            self.state["shared"] = self._projection(read_overlay(local.path))
            self.state["revision_transaction"] = self.state["decision"]["transaction"]
        self.conflicts = self._conflicts()
        if self.conflicts:
            LOG.warning("existing PC shared overlay requires adopt_shared/drop_shared: %s", self.conflicts)
        self._save()

    @property
    def publish(self):
        return self.local.publish

    @publish.setter
    def publish(self, callback):
        self.local.publish = callback

    def _projection(self, overlay):
        return {k: v for k, v in flatten(overlay).items() if k in self.keys}

    def _conflicts(self):
        if self.local.host != "pc":
            return {}
        actual = self._projection(read_overlay(self.local.path))
        known = self.state.get("shared", {})
        return {k: v for k, v in actual.items() if k not in known or known[k] != v}

    def _save(self):
        self.state["updated_ns"] = time.time_ns()
        self.state["keys"] = sorted(self.keys)
        self.state["conflicts"] = self._conflicts()
        atomic_write(self.state_path, json.dumps(self.state, sort_keys=True))

    def _call(self, request):
        if self.peer is None:
            raise TimeoutError("config peer offline")
        return self.peer.call(dict(request, type="ConfigCommand"), timeout=RPC_TIMEOUT_S)

    def _metadata(self, result):
        result["shared"] = dict(ready=self.state.get("ready", False),
                                conflicts=self._conflicts(), hosts=deepcopy(self.state.get("hosts", {})),
                                pending=bool(self.state.get("decision") or self.state.get("prepared")))
        return result

    def _plan(self, request):
        old = read_overlay(self.local.path)
        plan = self.local._handle(request, prepare_only=True)
        if "old" not in plan:  # no-op
            plan = dict(old=old, new=old, old_digest=digest(old), resulting_digest=digest(old),
                        overlay_keys=[], affected={}, actor=request["actor"], notice=None,
                        transaction=uuid.uuid4().hex)
        return plan

    def handle(self, request, *, forwarded=False):
        try:
            if (not isinstance(request, dict) or request.get("phase") or request.get("action") == "install"
                    or (not forwarded and {"pc_candidate", "pc_digest", "adopt"} & request.keys())):
                raise ConfigError("transaction phases are accepted only on the dedicated config link")
            # Forward outside the writer lock: coordinator prepare needs this host.
            if self.local.host == "pc":
                with self.lock, writer_lock(self.local.path):
                    old = read_overlay(self.local.path)
                    if request.get("expected_digest", digest(old)) != digest(old):
                        raise ConfigError("overlay changed; refresh and retry")
                    action = request.get("action")
                    if action == "adopt_shared":
                        values = self._projection(old)
                        if not values:
                            raise ConfigError("no PC shared overlay to adopt")
                        forwarded = dict(request, action="set_many", values=values,
                                         unset_keys=[], pc_digest=digest(old), adopt=True, pc_candidate=old)
                    elif action == "drop_shared":
                        return self._drop(request)
                    elif action in ("get", "status", "diff", "fold", "calibration_save"):
                        return self._metadata(self.local._handle(request))
                    else:
                        plan = self._plan(request)
                        if not (set(plan["overlay_keys"]) & self.keys):
                            if self.state.get("prepared"):
                                raise ConfigError("shared transaction prepared; retry after resolution")
                            return self._metadata(self.local.apply_plan(plan))
                        if self._conflicts():
                            raise ConfigError("existing PC shared overlay: adopt_shared or drop_shared first")
                        if not self.state.get("ready"):
                            raise ConfigError("PC shared configuration is not reconciled")
                        forwarded = dict(request, action="set_many",
                            values={k: flatten(plan["new"])[k] for k in plan["overlay_keys"]
                                    if k in self.keys and k in flatten(plan["new"])},
                            unset_keys=[k for k in plan["overlay_keys"] if k in self.keys and k not in flatten(plan["new"])],
                            pc_digest=digest(old), pc_candidate=plan["new"])
                expected = self.state.get("hosts", {}).get("jetson", {}).get("digest")
                if expected is None:
                    snapshot = self._call(dict(phase="snapshot", actor=request["actor"]))
                    if not snapshot.get("ok"):
                        return self._metadata(snapshot)
                    expected = snapshot["digest"]
                forwarded["expected_digest"] = expected
                result = self._call(dict(forwarded, phase="forward"))
                if result.get("ok"):
                    with self.lock:
                        self.state["hosts"] = deepcopy(result.get("hosts", self.state.get("hosts", {})))
                        self._save()
                # The caller's next expected digest belongs to PC, not Jetson.
                result["coordinator_digest"] = result.get("digest")
                result["digest"] = digest(read_overlay(self.local.path))
                return self._metadata(result)
            with self.lock, writer_lock(self.local.path):
                if request.get("action") in ("get", "status", "diff", "fold", "calibration_save"):
                    result = self.local._handle(request)
                    for name, host in self.state.get("hosts", {}).items():
                        if name != self.local.host:
                            result.setdefault("units", {}).update(host.get("units", {}))
                    return self._metadata(result)
                if self.state.get("decision"):
                    raise ConfigError("shared commit pending; wait for PC reconciliation")
                plan = self._plan(request)
                if not (set(plan["overlay_keys"]) & self.keys) and "pc_candidate" not in request:
                    return self._metadata(self.local.apply_plan(plan))
                return self._coordinate(plan, request)
        except (ValueError, KeyError, TypeError, OSError, TimeoutError) as exc:
            return self._metadata(dict(type="ConfigReply", ok=False, error=str(exc)))

    def _coordinate(self, plan, request):
        transaction = plan["transaction"]
        shared = self._projection(plan["new"])
        peer_online = self.peer is not None and self.peer.online()
        remote = None
        start = time.perf_counter()
        remote_ms = 0.
        if peer_online:
            if self.state.get("hosts", {}).get("pc", {}).get("state") != "ready" and not request.get("adopt"):
                raise ConfigError("PC online but shared configuration is unreconciled")
            expected = request.get("pc_digest", self.state.get("hosts", {}).get("pc", {}).get("digest"))
            rpc_start = time.perf_counter()
            try:
                remote = self._call(dict(phase="prepare", actor=plan["actor"], transaction=transaction,
                                         shared=shared, expected_digest=expected, adopt=request.get("adopt", False),
                                         pc_candidate=request.get("pc_candidate")))
            except (OSError, TimeoutError) as exc:
                self.state.setdefault("hosts", {})["pc"] = dict(state="prepare_unknown", error=str(exc))
                self._save()
                self.local.journal(dict(type="shared_prepare_timeout", transaction=transaction,
                                        hosts={"pc": self.state["hosts"]["pc"]}))
                raise
            remote_ms += (time.perf_counter() - rpc_start) * 1000
            if not remote.get("ok"):
                self.local.journal(dict(type="shared_rejected", transaction=transaction, hosts={"pc": remote}))
                return remote
        # A slow remote prepare must not turn an old off sample into authority.
        try:
            plan["notice"] = self.local.fire_control_notice(plan["overlay_keys"])
        except ConfigError:
            if remote is not None:
                try:
                    self._call(dict(phase="abort", actor=plan["actor"], transaction=transaction))
                except (OSError, TimeoutError):
                    pass  # reconciliation resolves an uncertain prepare
                self.state.setdefault("hosts", {})["pc"] = dict(state="reconcile_pending")
                self._save()
            raise
        # The decision precedes publication. Recovery repeats this idempotently.
        self.state["decision"] = dict(transaction=transaction, plan=plan, remote=remote, shared=shared)
        self._save()
        result = self.local.apply_plan(plan)
        self.state["shared"] = shared
        self.state["revision_transaction"] = transaction
        hosts = {"jetson": dict(state="committed", digest=result["digest"], units=result["units"], notice=result.get("notice"))}
        if remote is not None:
            try:
                rpc_start = time.perf_counter()
                response = self._call(dict(phase="commit", actor=plan["actor"], transaction=transaction,
                    coordinator=dict(state="committed", digest=result["digest"], units=result["units"], notice=result.get("notice"))))
                remote_ms += (time.perf_counter() - rpc_start) * 1000
                if not response.get("ok"):
                    raise ConfigError(response.get("error", "PC commit pending"))
                hosts["pc"] = dict(state="ready", digest=response["digest"], units=response["units"], notice=response.get("notice"))
                result["units"].update(response["units"])
                self.state["decision"] = None
            except (OSError, TimeoutError, ValueError) as exc:
                hosts["pc"] = dict(state="commit_pending", error=str(exc))
        else:
            hosts["pc"] = dict(state="pending", message="PC pending")
            self.state["decision"] = None
        self.state["hosts"], self.state["ready"] = hosts, True
        self._save()
        result.update(hosts=hosts, pc_round_trip_ms=remote_ms, coordination_ms=(time.perf_counter() - start) * 1000)
        self.local.journal(dict(type="shared_result", transaction=transaction, hosts=hosts))
        return self._metadata(result)

    def receive(self, request):
        """Dedicated-link entry point; never a PcModeTask or PC_ACTION."""
        try:
            phase = request.get("phase")
            if self.local.host == "jetson" and phase == "forward":
                forwarded = dict(request)
                forwarded.pop("phase")
                # Jetson's digest comes from Jetson, PC's from the forwarding host.
                return self.handle(forwarded, forwarded=True)
            with self.lock, writer_lock(self.local.path):
                if self.local.host == "jetson":
                    if phase == "snapshot":
                        decision = self.state.get("decision")
                        if decision:
                            plan = decision["plan"]
                            actual = digest(read_overlay(self.local.path))
                            if actual == plan["old_digest"]:
                                self.local.apply_plan(plan)
                            elif actual != plan["resulting_digest"]:
                                raise ConfigError("overlay differs from durable shared decision; manual recovery required")
                        return dict(type="ConfigReply", ok=True,
                                    shared=self._projection(read_overlay(self.local.path)),
                                    transaction=self.state["revision_transaction"], shared_digest=digest(self._projection(read_overlay(self.local.path))),
                                    hosts=deepcopy(self.state.get("hosts", {})), digest=digest(read_overlay(self.local.path)),
                                    decision=None if decision is None else dict(transaction=decision["transaction"]))
                    if phase == "report":
                        host = request["result"]
                        if host.get("state") == "ready" and host.get("shared_digest") != digest(self._projection(read_overlay(self.local.path))):
                            host = dict(host, state="mismatch", error="Jetson shared overlay changed during reconciliation")
                        self.state.setdefault("hosts", {})["pc"] = host
                        self.state["hosts"]["jetson"] = dict(state="ready", digest=digest(read_overlay(self.local.path)),
                                                             units=self.local.status())
                        if host.get("state") == "ready":
                            self.state["decision"] = None
                            self.state["ready"] = True
                            self.state["shared"] = self._projection(read_overlay(self.local.path))
                        self._save()
                        self.local.journal(dict(type="shared_receipt", transaction=request["transaction"], hosts={"pc": host}))
                        return dict(type="ConfigReply", ok=True)
                    raise ConfigError("invalid coordinator phase")
                transaction = request["transaction"]
                prepared = self.state.get("prepared")
                if phase == "prepare":
                    if prepared:
                        if prepared["transaction"] == transaction:
                            return dict(type="ConfigReply", ok=True, digest=prepared["old_digest"])
                        raise ConfigError("another shared transaction is prepared")
                    if self._conflicts() and not request.get("adopt"):
                        raise ConfigError("existing PC shared overlay requires adopt_shared/drop_shared")
                    old = read_overlay(self.local.path)
                    candidate = deepcopy(old if request.get("pc_candidate") is None else request["pc_candidate"])
                    for key in self.keys:
                        remove(candidate, key)
                    for key, value in request["shared"].items():
                        if key not in self.keys:
                            raise ConfigError("nonshared key in config link")
                        put(candidate, key, value)
                    plan = self._plan(dict(actor=request["actor"], action="install", candidate=candidate,
                                           expected_digest=request["expected_digest"]))
                    plan["transaction"] = transaction
                    self.state["prepared"], self.state["ready"] = plan, False
                    self._save()
                    self.local.journal(dict(type="shared_prepared", transaction=transaction,
                                            resulting_digest=plan["resulting_digest"], units=plan["affected"]))
                    return dict(type="ConfigReply", ok=True, digest=plan["old_digest"], units=plan["affected"])
                if phase == "abort":
                    if prepared and prepared["transaction"] == transaction:
                        self.state["prepared"] = None
                        self._save()
                    return dict(type="ConfigReply", ok=True)
                if phase == "commit":
                    if (not prepared and self.state.get("last_transaction") == transaction
                            and digest(read_overlay(self.local.path)) == self.state["last_result"]["digest"]):
                        return self.state["last_result"]
                    if not prepared or prepared["transaction"] != transaction:
                        raise ConfigError("no matching prepared transaction")
                    actual = digest(read_overlay(self.local.path))
                    if actual == prepared["resulting_digest"] and actual != prepared["old_digest"]:
                        result = dict(type="ConfigReply", ok=True, digest=actual, units=self.local.status(),
                                      transaction=transaction, recovered=True)
                    elif actual == prepared["old_digest"]:
                        result = self.local.apply_plan(prepared)
                    else:
                        raise ConfigError("PC overlay changed after prepare; manual recovery required")
                    self.state.update(prepared=None, shared=self._projection(prepared["new"]), ready=True,
                                      last_transaction=transaction, last_result=result)
                    self._save()
                    self.local.journal(dict(type="shared_result", transaction=transaction,
                        hosts={"pc": dict(state="ready", digest=result["digest"], units=result["units"]),
                               "jetson": request.get("coordinator", dict(state="committed"))}))
                    return result
                raise ConfigError("invalid participant phase")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return dict(type="ConfigReply", ok=False, error=str(exc), host=self.local.host,
                        transaction=request.get("transaction"), phase=request.get("phase"))

    def _drop(self, request):
        if self.state.get("prepared"):
            raise ConfigError("shared transaction prepared; resolve before dropping")
        candidate = read_overlay(self.local.path)
        for key in self.keys:
            remove(candidate, key)
        plan = self._plan(dict(actor=request["actor"], action="install", candidate=candidate,
                               expected_digest=request.get("expected_digest", digest(read_overlay(self.local.path)))))
        result = self.local.apply_plan(plan)
        self.state.update(shared={}, ready=False)
        self._save()
        return self._metadata(result)

    def reconcile(self):
        """PC calls this off its owning loop before it advertises readiness."""
        transaction = uuid.uuid4().hex
        initial_prepared = self.state.get("prepared")
        initial_transaction = None if initial_prepared is None else initial_prepared["transaction"]
        try:
            snapshot = self._call(dict(phase="snapshot", actor="config-reconcile"))
            if not snapshot.get("ok"):
                raise ConfigError(snapshot.get("error", "coordinator unavailable"))
            decision = snapshot.get("decision")
            if decision:
                transaction = decision["transaction"]
                result = self.receive(dict(phase="commit", actor="config-reconcile", transaction=transaction,
                                           coordinator=snapshot.get("hosts", {}).get("jetson", dict(state="committed"))))
            elif (self.state.get("ready") and not self.state.get("prepared") and not self._conflicts()
                  and self._projection(read_overlay(self.local.path)) == snapshot["shared"]):
                transaction = snapshot["transaction"]
                result = dict(ok=True, units=self.local.status())
            else:
                prepared = self.state.get("prepared")
                if prepared:
                    if prepared["transaction"] != initial_transaction:
                        return dict(state="pending", error="coordinator prepare in progress")
                    self.receive(dict(phase="abort", transaction=prepared["transaction"]))
                transaction = snapshot["transaction"]
                result = self.receive(dict(phase="prepare", actor="config-reconcile", transaction=transaction,
                                           shared=snapshot["shared"], expected_digest=digest(read_overlay(self.local.path))))
                if result.get("ok"):
                    result = self.receive(dict(phase="commit", actor="config-reconcile", transaction=transaction,
                                           coordinator=snapshot.get("hosts", {}).get("jetson", dict(state="committed"))))
            host = dict(state="ready" if result.get("ok") else "rejected",
                        digest=digest(read_overlay(self.local.path)), shared_digest=digest(self._projection(read_overlay(self.local.path))),
                        units=result.get("units", {}),
                        error=result.get("error"))
            self._call(dict(phase="report", actor="config-reconcile", transaction=transaction, result=host))
            with self.lock:
                self.state["ready"] = host["state"] == "ready"
                self.state["hosts"] = {"pc": host, "jetson": dict(state="ready", digest=snapshot["digest"],
                    units=snapshot.get("hosts", {}).get("jetson", {}).get("units", {}))}
                self._save()
            return host
        except (ValueError, KeyError, OSError, TimeoutError) as exc:
            with self.lock:
                self.state["ready"] = False
                self._save()
            return dict(state="pending", error=str(exc))


class ConfigLink:
    """Full duplex bounded RPC; PC connects, Jetson binds. No inbound PC port."""
    def __init__(self, context, endpoint, agent, *, server=False):
        self.context, self.endpoint, self.agent, self.server = context, endpoint, agent, server
        self.out = queue.Queue(maxsize=64)
        self.pending = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.seen = 0.
        self.route = None
        self.handlers = threading.BoundedSemaphore(32)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="config-link-handler")
        self.thread = threading.Thread(target=self._run, name="config-link-io", daemon=True)
        agent.peer = self
        self.thread.start()
        self.reconciler = None
        if not server:
            self.reconciler = threading.Thread(target=self._reconcile, name="config-reconcile", daemon=True)
            self.reconciler.start()

    def online(self):
        return time.monotonic() - self.seen < 1.5

    def call(self, request, *, timeout=RPC_TIMEOUT_S):
        token, response = uuid.uuid4().hex, queue.Queue(maxsize=1)
        with self.lock:
            self.pending[token] = response
        try:
            self.out.put_nowait(dict(id=token, request=request))
            try:
                return response.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError("config peer timed out")
        finally:
            with self.lock:
                self.pending.pop(token, None)

    def _respond(self, token, request):
        try:
            response = self.agent.receive(request)
        except Exception as exc:
            LOG.exception("config link handler failed")
            response = dict(type="ConfigReply", ok=False, error=str(exc))
        try:
            self.out.put_nowait(dict(reply_to=token, response=response))
        except queue.Full:
            pass
        finally:
            self.handlers.release()

    def _run(self):
        socket = self.context.socket(zmq.ROUTER if self.server else zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 64)
        if self.server:
            socket.bind(self.endpoint)
        else:
            socket.connect(self.endpoint)
        heartbeat = 0.
        try:
            while not self.stop.is_set():
                now = time.monotonic()
                if not self.server and now >= heartbeat:
                    try:
                        socket.send_json(dict(heartbeat=True), flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
                    heartbeat = now + .25
                if socket.poll(10):
                    frames = socket.recv_multipart()
                    if self.server:
                        self.route = frames[0]
                    try:
                        packet = json.loads(frames[-1])
                        self.seen = time.monotonic()
                        if packet.get("heartbeat"):
                            if self.server:
                                socket.send_multipart([self.route, b'{"heartbeat":true}'], flags=zmq.NOBLOCK)
                        elif "reply_to" in packet:
                            with self.lock:
                                waiter = self.pending.get(packet["reply_to"])
                            if waiter is not None:
                                waiter.put_nowait(packet["response"])
                        else:
                            if self.handlers.acquire(blocking=False):
                                self.pool.submit(self._respond, packet["id"], packet["request"])
                            else:
                                self.out.put_nowait(dict(reply_to=packet["id"], response=dict(
                                    type="ConfigReply", ok=False, error="config link worker busy")))
                    except (ValueError, KeyError, queue.Full, zmq.Again):
                        LOG.warning("invalid or congested config-link packet")
                for _ in range(16):
                    try:
                        packet = self.out.get_nowait()
                    except queue.Empty:
                        break
                    if self.server and self.route is None:
                        continue
                    frames = [json.dumps(packet).encode()]
                    if self.server:
                        frames.insert(0, self.route)
                    try:
                        socket.send_multipart(frames, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
        finally:
            socket.close(0)

    def _reconcile(self):
        while not self.stop.wait(.5):
            if self.online():
                self.agent.reconcile()
            else:
                with self.agent.lock:
                    if self.agent.state.get("ready"):
                        self.agent.state["ready"] = False
                        self.agent._save()

    def close(self):
        self.stop.set()
        if self.reconciler:
            self.reconciler.join(2 * RPC_TIMEOUT_S + 1)
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.thread.join(2)

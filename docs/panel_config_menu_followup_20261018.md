# Panel Settings responsiveness follow-up, 2026-10-18

Base: unchanged `0550f2b` (`lab/panel-config-menu`). Work is on
`codex/panel-config-menu-followup`. No deployment or lab-host access.

## Implementation

`common/live_config.py`: one config worker per `AgentChannel` serializes both
local CLI requests and forwarded Jetson panel requests. The existing
`writer_lock`, expected digest, full validation, journal intent/result order,
restart allow-list and motor/engagement vetoes remain in place. Host-local REP
becomes ROUTER. `jetson/operator_agent.py` uses ROUTER on its existing operator
endpoint too, preserving the entire REQ routing envelope, including optional
correlation IDs. Ordinary operator replies remain immediate while config replies
are deferred. No request or reply JSON fields change.

The worker performs validation, durable writes, receipt waits and restart work.
It queues publications and replies in order. The owning loop alone operates the
ROUTER/PUB sockets and calls the LiveSettings safe point. A worker-owned PUSH
socket wakes an owner-owned PULL socket in the operator poller when work finishes;
fast cached replies do not wait for the idle 50 ms poll timeout. Shutdown joins
the worker before closing its owner's sockets. The request queue is bounded at
32, with a busy rejection on overflow. `pc.mode_agent` uses the same updated
channel, so its config work also leaves the PC polling thread. Its pre-existing
outbound network timeouts and 0.2 s polling period remain unchanged.

`ConfigAgent.bundles` caches validated immutable bundles. Flattened values and
controller parser defaults are cached separately and copied for replies. The
cache key combines the overlay SHA-256 digest with an ordered stack fingerprint:
unit, module and every stack file's resolved path, size and `mtime_ns`. Shared
files are statted/resolved once per fingerprint. A changed file fingerprint
clears both caches; a different overlay digest cannot use the previous overlay's
entry. The bounded cache keeps release plus two recent overlay/candidate entries.
Builds spanning file edits fail closed, including edits between old/candidate
validation. A warm mutation validates only its candidate. `status` does no stack
or parser rebuild, and always collects current receipts.

Existing `pi-panel` and `pi-panel:*` actors skip receipt collection/wait in mutation
handling; CLI actors retain the previous behavior, including its immediate
receipt check when configured with a zero wait. Panel replies initially describe
active hot consumers as `awaiting_service`; inactive/restart/veto states retain
their meanings. Publication and the final journal record precede the reply.

`rpi/display/config_menu.py` polls `status` rather than `get` while root/group/result
is open. Open, digest changes, mutation replies and explicit Refresh load full
values. Matching fresh status receipts advance the existing result display;
stale or other-digest receipts cannot claim application. Accepted menu contents,
steps/bands, authority and fire-control guards, calibration/set_many behavior,
tiers and schema are unchanged.

## Local evidence

Windows 10 build 19045, Python 3.14.4, pytest 9.0.3, pyzmq 27.1.0, PyYAML 6.0.3,
pydantic 2.13.3. Inputs are the eight versioned Jetson stacks and deterministic
fake service/IPC clients. No learned model, GPU, network host, motor or serial
interface is exercised. Timing clocks are local `perf_counter`/`monotonic`.

The latest focused run measured cached eight-stack `ConfigAgent.handle(get)`
over ten requests at **12.37 ms median / 12.46 ms maximum**, and a warm panel
`handle(set operator.lock_lost_s)` at **408.82 ms**, including real temporary
overlay/journal writes with fake service state. These are handler timings, not
CLI process startup/preflight or network round-trip measurements.

With each `bundles` call deliberately sleeping **1 second**, the production
operator loop's largest selection publication gaps were **0.252 s** through the
panel endpoint and **0.251 s** through the CLI endpoint. Simultaneous operator
release/status replies stayed at or below **0.003 s**, and PC poll replies at or
below **0.002 s**. Lock-expiry ticks continued. The test asserts gaps/replies no
greater than `2 * SELECTION_PERIOD_S` (0.4 s) and uses bounded client lifetimes.
REQ correlation envelopes, FIFO writes/stale-digest rejection, publication-before-
reply ownership and worker wakeups are also covered.

Commands and final counts are recorded in the migration journal. Raw logs,
temporary overlays and release-bundle/runtime provenance are retained under
ignored `.task-tmp/followup-*`; `followup-provenance.json` records all eight
resolved release digests and the dirty candidate's base commit. No source config
or model files changed. Full-suite limitations are missing Pillow/trimesh,
Windows-unavailable `gi`/GStreamer, and the absent external gimbal fit report.
The initial default-temp run also exposed case-sensitive path assertions on
Windows; the unchanged preflight tests pass with a workspace-local basetemp.

## Reviewer commands and acceptance

Run in the reviewed Jetson checkout using its existing environment. These tests
use isolated temporary overlays, fake service runners and in-process IPC; they
do not need active GPU/video/serial units and do not activate any unit:

```sh
bin/python3 -m pytest -q -s tests/test_config_agent_responsiveness.py
bin/python3 -m pytest -q tests/test_panel_config_menu.py tests/test_live_config.py \
  tests/test_operator_agent.py tests/test_pc_mode.py tests/test_operator_display.py \
  tests/test_operator_menu.py
```

Acceptance: all applicable contracts pass; cached handler `get` is well below
**50 ms**, warm hot mutation reply below **0.8 s**, and slow-work selection gaps
at most **0.4 s** for both entry points. The first command prints those timings.
Do not include CLI startup and its independent local preflight in handler timing.
Restart latency is service-dependent and is not qualified by the hot-key test.

For the subsequently reviewed installed software candidate, follow the existing
owner/bench gates in `launch_procedure.md` and `operator_display.md`; do not start
inactive units for this check. From that release, a passive selection watcher is:

```sh
bin/python3 - <<'PY'
import time
from pathlib import Path
import zmq
from common.live_config import runtime_bundle, unit_stacks
from jetson.operator_agent import AgentConfig
spec = unit_stacks(Path.cwd(), 'jetson')['jetson/idcs-operator-agent.service']
cfg = AgentConfig.from_config(runtime_bundle(spec['paths']).data)
endpoint = cfg.selection_bind.replace('tcp://*:', 'tcp://127.0.0.1:')
times = []
with zmq.Context() as ctx:
    with ctx.socket(zmq.SUB) as sub:
        sub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt_string(zmq.SUBSCRIBE, '')
        sub.connect(endpoint)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if sub.poll(200):
                sub.recv()
                times.append(time.monotonic())
gaps = [b-a for a, b in zip(times, times[1:])]
assert len(times) > 100, f'insufficient selection traffic: {len(times)}'
print('selection max receive gap, seconds:', max(gaps))
assert max(gaps) <= .4
PY
```

During the watcher, keep Settings open for several status polls, perform one
reviewed hot `operator.lock_lost_s` edit and restore its previous value. Require
no gap above 0.4 s throughout, no mutation timeout, initial awaiting-service
followed by a fresh matching receipt, and responsive ordinary operator commands
and PC mode polls. Measure mutation round-trip below 0.8 s on that host as well.
This passive receive-time measurement includes local scheduling/transport and
therefore complements the isolated publish-call timing. Installed-host timing,
Pi rendering and bench usability remain reviewer work; no hardware actuation
qualification is claimed.

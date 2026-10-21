# Resident runtime service migration plan

Status: proposed implementation plan, 2026-10-21. Baseline: deployed code
`d0e9502` / deploy-111; documentation receipt `b6b59a8`. This plan does not
describe an already implemented replacement or authorize hardware activation.

## Outcome and scope

Replace mode-dependent process lifetimes with persistent process owners and
acknowledged runtime state. Preserve the resident YOLO/ROI switch already
implemented in deploy-111. Extend that approach to video source, SIM scene,
motion backend, controller settings, viewers and recording.

The target is 13 service roles: Jetson6, PC5, Pi2, replacing the current26
managed services. This is a functional ownership decision, not a requirement
to merge unrelated work merely to reach a number. Systemd retains startup,
crash isolation, shutdown and immutable-release deployment. Ordinary operator
changes do not invoke systemctl. Optional inputs can sleep inside their owner;
resident does not mean continuously running both sources at maximum load.

Hardware activation remains prohibited until the repaired-wiring/all-required-
motors-answer gate in AGENTS.md is satisfied and the user authorizes bench work.
Software development and deterministic fake-backend validation proceed first.
The existing physical bridge/serial units must not be started as a shortcut
to testing the new software backend.

## Design decisions

1. Jetson owns the controller and bridge for both simulated and real motion.
   There is no separate operational PC controller. PC renders from the selected
   backend's qualified pose; it does not integrate another hidden mount.
2. One bridge exposes a simulated backend, real backend and disabled state.
   Shared authority/freshness/limits precede backend-specific translation.
   Fake mode does not open serial, instantiate physical sensor readers, issue
   initialization commands or depend on the physical serial service starting.
3. Simulated dynamics and gains are an independent stable substitute. Physical
   F6 quantization, gearing, bench gains and parameter writes belong to the real
   adapter. Optional identified plants and packet-level motor emulation remain
   explicit diagnostic profiles, not the default visual-SIM model.
4. One Jetson perception process owns camera/SIM inputs, YOLO/NvDCF or ROI,
   selection, source identity, OSD and a shared return encoder.
5. One PC renderer and one PC viewer survive source/scene changes. The Pi's
   existing persistent display and receiver-only recovery are retained.
6. Video source, tracking method, motion backend and panel authority are
   separate dimensions. Selecting a source or processing mode is not arming.
7. Source identity and hardware authority never come from whether a systemd
   unit is active. Use progressing typed state and applied-operation receipts.
8. Source switches preserve video continuity where qualified; motion holds
   while required measurements/coordinates are invalid. Do not promise zero
   acquisition time, zero handover time or seamless cold model replacement.

## Final service ownership

Names below are the proposed installed names. Existing implementations are
reused/extracted; the table does not imply13 new daemons in addition to old ones.

### Jetson system services

| Service | Sole responsibility | Resident behavior |
|---|---|---|
| `idcs-perception.service` | Camera/SIM ingest, source identity, YOLO/tracker, resident VPI ROI, target assessments/selection, V2 publication, GPU OSD, return encoder/RTCP | Changes input/processing through bounded commands, without process replacement |
| `idcs-controller.service` | Atomic observations, panel authority, basic PID, separate Kalman feedforward/prediction, manual slew, coasting/idle return, short-lease intents and diagnostics | Explicit SIM/hardware/shadow profiles; no serial access |
| `idcs-bridge.service` | Common intent gate, exclusive backend ownership, command application and qualified pose/applied-control feedback | Simulated/real/disabled backend handover; never automatically selects real |
| `idcs-serial.service` | Sole physical RS485 owner, priority scheduling, wire dispatch, replies, terminal execution evidence | Real backend only; absent/inactive in software-only rollout; later authorized real sessions need not restart it for video/scene changes |
| `idcs-operator-agent.service` | Authoritative desired/applied runtime state, coordinated transitions, target/operator requests, config coordination and recording status | Survives every operational transition; does not process video or run PID |
| `idcs-recorder.service` | Read-only Jetson flight/control/perception/transition records with provenance, rotation and quotas | Opens/closes recording sessions without stopping its process |

The new fake-capable bridge must not have an unconditional Requires relationship
that starts serial. Moving the process boundary does not grant the existing
physical units permission to start. During migration, exercise the extracted
software runtime under isolated test commands, not the physical service names.

### PC user services

| Service | Sole responsibility | Resident behavior |
|---|---|---|
| `idcs-sim-runtime.service` | OpenGL mesh renderer, Single/Swarm scene state, target truth for scoring only, selected-backend pose consumption, verified RTP uplink, source-clock responder | Scene and pose-source changes at frame boundaries; no controller or independent operational plant |
| `idcs-ui.service` | Read-only return-video viewer, local receiver recovery and receiver-owned HUD | One viewer for Camera, software SIM and HIL; no motion authority |
| `idcs-pc-mode-agent.service` | PC-initiated coordinator connection, runtime command forwarding/receipts, session/capability registration, shared-config participant | Runtime commands replace scene/source service operations; retain outbound connection model |
| `idcs-hud-relay.service` | Read-only SIM scene/score plus authoritative Jetson pose/control telemetry to Pi | Relays original provenance and accumulated age; never republishes truth as visual/control evidence |
| `idcs-pc-recorder.service` | PC rendering, scene truth, pose-consumption and transport records | Recording sessions independent of source/scene process lifetime |

Keeping the HUD relay separate preserves its display-only boundary and isolates
network/display faults from rendering. Embedding the existing clock responder
in the resident source owner does not create a separate service.

### Pi user services

| Service | Sole responsibility | Resident behavior |
|---|---|---|
| `idcs-manual.service` | Physical GPIO/ADC inputs, joystick, panel state, lamp policy and applied-status indication | Retain existing name for compatibility; it is the full panel service, not only manual control |
| `idcs-operator-display.service` | Return receiver/recording tap, HUD, mandatory console strip, menus, target picker and exact-frame ROI editor | Persistent application with receiver-only recovery; sends requests, never motor commands |

Pi return-video recording remains inside its existing receiver tap, not another
decoder, encoder or service. Offline analysis/training, fitting, benchmarks and
maintenance tools remain commands, not permanent service roles.

## Mapping every existing service

### PC fourteen to five

| Current service | Destination or disposition |
|---|---|
| `idcs-camera-ui` | Unified `idcs-ui` Camera view |
| `idcs-hil-streamer` | `idcs-sim-runtime`, real measured-pose mode |
| `idcs-hud-relay` | Retained; consumes tagged Jetson state and PC scene/score |
| `idcs-pc-mode-agent` | Retained; runtime protocol replaces ordinary systemctl actions |
| `idcs-sim-controller` | Jetson resident controller with independent stable-SIM profile |
| `idcs-sim-panel` | Jetson controller's panel-source adapter; real Pi state first, explicit fake-only panel-less test facility; no separate PC publisher |
| `idcs-sim-recorder` | Generalized/renamed `idcs-pc-recorder` |
| `idcs-sim-streamer` | `idcs-sim-runtime`, Single scene |
| `idcs-sim-swarm-controller` | Same Jetson resident controller, visual V2 input |
| `idcs-sim-swarm-streamer` | `idcs-sim-runtime`, Swarm scene |
| `idcs-sim-ui` | Unified `idcs-ui` SIM view |
| `idcs-sim-visual-lock-controller` | Jetson controller; isolated legacy verifiers remain tools |
| `idcs-sim-visual-lock-streamer` | Renderer scene/fixture preset, not another installed streamer |
| `idcs-ui` | Reworked from HIL-only viewer into unified read-only viewer |

### Jetson ten to six

| Current service | Destination or disposition |
|---|---|
| `idcs-deepstream-camera` | Camera input adapter of `idcs-perception` |
| `idcs-deepstream-video` | Verified RTP input adapter of `idcs-perception` |
| `idcs-controller-camera` | Resident controller, Camera timing and shadow execution policy |
| `idcs-controller` | Resident controller, selected backend/profile; no separate camera/HIL instances |
| `idcs-bridge` | Retained with extracted simulated/real backend interfaces |
| `idcs-serial` | Retained as exclusive real bus owner |
| `idcs-operator-agent` | Retained as runtime coordinator |
| `idcs-recorder` | Retained with resident recording-session commands |
| `idcs-visual-lock-camera` | Normal feature already resides in perception; preserve standalone native verifier/fallback source as an exclusive maintenance tool, retire ordinary installed unit after parity |
| `idcs-visual-lock-sim` | Same; no second operational GPU pipeline |

### Pi two to two

Both existing services retain their ownership. Update subscriptions, runtime
generation handling, menus and recording receipts, not the GPIO wiring map or
physical meaning of the switches.

### Existing groups and final groups

| Existing target | Replacement |
|---|---|
| PC `idcs-sim` | Runtime preset: SIM source, Single scene, simulated backend |
| PC `idcs-sim-swarm` | Runtime preset: SIM source, Swarm scene, simulated backend |
| PC `idcs-sim-visual-lock` | SIM fixture preset plus explicit ROI processing; standalone fallback remains maintenance-only |
| PC `idcs-sim-session` | Persistent owners with an acknowledged runtime session; no viewer/controller/panel restart cascade |
| Jetson `idcs-camera` | Camera source plus shadow/disabled execution preset |
| Jetson `idcs-hil` | SIM source plus real backend preset, subject to separate bench authorization |

Use one optional `idcs-runtime.target` per host to start that host's ordinary
resident software owners. These are three host-local grouping instances, not
a cross-host atomic start and not three more processes. Jetson software startup
must not pull in serial or real activation. Systemd dependency ordering keeps
the real bus available while the bridge performs an authorized stop/release.
Do not create a separate operational target for every state combination.

## Runtime state and supported combinations

The coordinator owns a validated desired state and separately reports each
owner's applied state. Proposed fields:

- Video source: `none`, `camera`, `sim`.
- SIM scene: `single`, `swarm`; fixture/replay presets are maintenance inputs.
- Processing: `yolo`, `roi`; detector/tracker implementation remains separately
  configured, not silently changed by selecting a motion backend.
- Motion backend: `disabled`, `simulated`, `real`.
- Execution: `shadow` or `active`, further constrained by real permission.
- Selection policy: existing `planner` or `operator`, source/session scoped.
- Control authority: disabled/manual/auto, derived from fresh panel inputs and
  safety gates, not set as a shortcut by a menu preset.
- Profile/calibration digests, source session, tracking revision, backend epoch,
  configuration revision, recording session and transition ID.

| Operator preset | Video and scene | Motion and execution | Interpretation |
|---|---|---|---|
| Standby | None | Disabled | Persistent UI/control plane; no valid target or applied motion |
| Software SIM | SIM Single/Swarm | Simulated, active when panel permits | Full visual/control/bridge loop against stable substitute |
| Camera preview | Camera | Disabled/shadow | Video/selection and qualified diagnostic calculation; no real movement |
| Camera operation | Camera | Real, active only with physical permission | Missing current operational combination becomes a runtime preset, not another stack |
| SIM HIL | SIM Single/Swarm | Real, active only with physical permission | Renderer follows measured real pose; no synthetic fallback |
| Diagnostic replay | Explicit recorded/fixture source | Simulated or disabled | Maintenance-only, provenance labelled; cannot acquire real permission |

YOLO and ROI are available in Camera/SIM presets when their current capability
gates pass. Camera plus simulated motion can be an explicitly labelled diagnostic
experiment, but does not close a physical image-motion loop; do not present it
as successful camera tracking. Missing pose in shadow is missing evidence, not
a fabricated valid zero position.

`sim.use_jetson_cam_state` must not remain a second independent authority knob.
During compatibility, SIM+real derives true and SIM+simulated derives false;
contradictory legacy overlays are rejected. Camera behavior does not depend on
this SIM-only flag. Remove the legacy selector only after all consumers migrate.

## Preserve every operational feature

| Feature family | Final owner and preservation requirement |
|---|---|
| YOLO detection | Perception retains deployed engine/parser/labels, grey letterbox, threshold/class mapping and model digests; moving processes must not change preprocessing |
| NvDCF, optional NvSORT, persistence/re-ID | Perception retains configured tracker, coasting/shadow-track and identity-stitching rules; internal IDs remain internal |
| Resident ROI | Keep VPI CUDA DCF + Harris + forward/backward PyrLK, exact displayed-frame seed/cache, bounded catch-up, lease expiry, release/loss/refusal and fresh lock namespaces |
| ROI mode semantics | No automatic YOLO-target-to-ROI or VPI-to-NvDCF handover; explicit seed starts ROI; release/loss stays ROI; explicit YOLO resumes detection |
| ROI responsiveness | Keep fixed video path, bounded asynchronous native worker, pre-OSD NVMM copies, actual source/completion timestamps, visible worker faults and teardown ownership |
| Planner | Preserve learned and rule-based policies, asynchronous age bounds, rankings, assessments, exclusions and class eligibility; no learned-model dependence in downstream contract tests |
| Operator target choice | Preserve joystick cycling, frozen picker candidates, explicit lock/release and source-session validation; Planner OFF holds nearest eligible ranged target until lost, then reacquires nearest after grace without falling back to Planner |
| Ranging | Preserve known-size geometry, class/aspect/border/jump guards, bounded held range with original acceptance age, class/session reset and no refreshed age from suspect boxes |
| Partial obstruction and parallax | Image tracking may continue while range is briefly held; expiry withdraws depth-dependent correction rather than inventing depth. Preserve configured camera-to-aim offset; arbitrary unknown-class ROI has no fabricated known size |
| Control | Keep atomic V2 observation, BasicPID feedback separated from Kalman target-rate FF/prediction, manual slew, source-time pose history, limits, diagnostics, zero intents, coast and idle return |
| Controller tuning | Separate immutable stable-SIM profile from bench/tuned hardware profile. Preserve offline plant fitting/gain-search and PID/FF A/B tools; no visual SIM result qualifies physical gains |
| Mount feedback | Real adapter preserves step-count/encoder anchoring, independent sample times, dual-pitch mapping/divergence handling and configured optional sensors; no IMU activation by migration |
| Motor transport | Serial remains sole physical owner; preserve timed commands, refresh/expiry constraints, emergency priority, latest-wins/stale filtering and dispatch/reply diagnostics |
| Panel inputs | Preserve GPIO12 automatic gate, independent GPIO5 manual authority, GPIO16 fire-button indication only, E-stop dominance and joystick calibration/deadzone |
| Shared lamp contacts | Only GPIO12/16 affected; unreadable is not OFF, permission fails closed, fresh post-power sampling/hysteresis remain; never latch a stale HIGH |
| Lamps | Safety means fresh applied simulated/real tracking or manual exception; fire-control means target available; yellow requires actual fresh physical motor feedback, never fake feedback; green/red meanings unchanged |
| ROI panel workflow | In Manual, readable OFF selects/edits and ON locks after the required edge; initial ON/unreadable edges do not seed. Resident lock survives Manual OFF; auto follows only through ordinary gates. Menu-only joystick inversion never changes pitch motion |
| Fire indication | Preserve armed/pressed/unknown HUD and recorded engage/refusal semantics; there is no effector, firing output or new actuation capability |
| GPU target OSD | Jetson owns accepted boxes, compact class/confidence/range, selected/coast/ROI states and solid rank-corner badge; rank only with multiple accepted targets, no displayed track ID |
| PC/Pi HUD | Shared theme/reticle colors/tapes/parallax/FF cue; no second target box or duplicate target facts. Receiver-owned telemetry is labelled latest, not an exact video join |
| Mandatory console | E-stop/fire/authority/recording remain in foreground above menus; preserve low-video-rate, stale/link-loss and recovery alerts |
| Swarm score and Single error | PC scene owns score/truth; controller owns measured raw tracking error. Preserve mutually exclusive metric strip and session-scoped confirmed score reset without resetting targets, recordings or locks |
| Video transport | Preserve verified uplink RTP identity, native return RTCP feedback/keyframe gating, independent PC/Pi recipients, bounded queues and receiver-only repair; never turn all frames into IDR as an unmeasured substitute |
| Recording | Preserve Jetson/PC chronological telemetry, Pi keyframe-aligned segmented return MKV without re-encoding, quotas, retained artifacts and REC status; clearly distinguish Jetson OSD from receiver-local HUD not burned into that file |
| Live configuration | Preserve strict schema, local/shared overlays, prepare/reconcile receipts, safe-point tiers, panel editing guards and release/config/model provenance; runtime mode changes become typed transactions rather than mutable overlay tricks |
| Health and diagnostics | Retain rates/ages/clock intervals/estimator terms/command outcomes in health and Information pages, not an intrusive permanent HUD stack; expose owner, selected backend and applied generation |
| Recovery/deployment | Preserve persistent Pi menu during receiver loss, process guards, exclusive GPU/bus ownership, immutable releases, staged validation, manifests, audit and rollback |

Keep VPI initialize-to-NvDCF experiments and MaskTracker/other algorithm research
outside the operational mode contract until independently qualified. Existing
training/export/evaluation and native lifetime/interop tools remain available.
The current operator command contract has class eligibility, not general manual
per-target threat/firing-eligibility overrides; do not claim such an override was
implemented merely by reorganizing the menu. Any later addition needs its own
typed policy command and must not create effector authority.

## Video and perception source handover

The resident pipeline has source-specific ingest/clock/geometry adapters feeding
a common canonical processing geometry and shared output. Preserve each source's
actual sensor/render timestamp and calibrated projection through resize/crop;
do not relabel arrival time or apply one source's intrinsics to the other.

Keep input buffers bounded and the inactive input from stalling streammux.
Do not feed both full-rate sources into expensive inference merely to keep
them warm. During preparation, warm capture/decoder and validate unique frames
on the destination without changing the active perception publication.

An implementation spike must qualify NvDCF state reset across source epochs.
Use a supported stream reset with empty output until fresh tracks, or isolated
warm per-source tracker state if measured resource limits require that design.
Never carry the old source's trackers/range cache through a persistent process
just because its process UUID did not change. The observable contract, not an
assumed plugin reset, is the acceptance condition.

The output encoder, return RTP session, OSD and Pi/PC receiver processes remain
stable during a qualified hot switch. Source/session identity changes inside
the sideband and V2 records even though return SSRC need not change. Retire exact
ROI return-frame joins from the old source and refuse late seeds/results.

Standby uses a clearly invalid/no-source presentation branch; black/generated
frames must not masquerade as verified camera frames. A retained frame can be
marked stale, never counted as a newly decoded/source-observed frame.

Cold startup, capture-device recovery, resolution/model/tracker replacement or
insufficient warm-source resources may need a bounded unavailable interval.
Report that explicitly. Hot-switch qualification cannot be claimed by hiding
a frozen frame or silently rebuilding the whole process.

## Bridge backend contract and ownership

Extract an interface equivalent to prepare, readiness, apply, sample, stop and
close. Each operation carries backend identity/epoch, command sequence, lease,
configuration/calibration digest and actual application/sample timestamps.
Admission, simulated application, physical wire dispatch, firmware reply and
mechanically verified motion remain distinct evidence levels.

The simulated backend integrates one versioned stable plant on a monotonic
clock and supplies pose/rate/application receipts. A fake clock supports replay.
The renderer consumes that pose rather than advancing a second plant. When
Jetson is late/disconnected, rendering marks pose stale; it does not quietly
restore the old PC integrator. Real HIL retains measured-pose timeline rules.

The real backend alone translates physical units/signs/gearing to timed MKS
commands, uses serial replies and owns initialization/anchoring/release.
Physical sensors are lazy, real-only dependencies. A protocol-emulating fake
serial peer can test wire/queue behavior, separately from ideal motion SIM.

Serial dispatch must enforce real authority fencing too. Checking epochs only
in the bridge is insufficient if old nonzero commands remain queued below it.
Extend serial IPC to fence/discard old motion at admission and immediately before
wire dispatch; distinguish zero/stop cleanup from new nonzero authority. Legacy
unfenced live command producers are refused on the production endpoint after
migration. Keep isolated bench clients explicit and mutually exclusive.

There is no real-to-fake or fake-to-real automatic fallback. The real port is
never opened because a simulator needs readiness. Yellow availability may use
authorized real read-only health, but fake pose/application never qualifies it.
Physical profile changes cannot be passed through the fake backend to hardware.

## Transition protocol and motion handover

One serialized transition owner coordinates the active participants. Typed
messages carry request ID, expected state revision, coordinator/process session,
affected dimensions, deadline and desired profile digests. Commands are
idempotent; duplicate requests return their recorded result, not another switch.
The existing ROI mode command remains its low-latency, session-scoped path;
coordinator state observes its applied revision instead of replacing it with
a slow service-management transaction.

| State | Required behavior |
|---|---|
| Running | Publish applied state and active owner health; request is not yet application proof |
| Preparing | Validate compatibility/capabilities and prepare destination while retaining usable old video; reject overlapping structural config/mode changes |
| Quiescing | Inhibit affected motion, cancel obsolete locks/leases and fence queued commands; real stop must be confirmed or its qualified expiry established |
| Committing | Issue a new affected generation and apply input/backend/profile at declared owner boundaries; no owner infers commitment from wall-clock arrival |
| Verifying | Require matching participant receipts, fresh unique source/pose data, timing bounds, calibration and panel state; controller output stays inhibited until valid |
| Running new state | Publish acknowledged state, resume only permitted operations and record the transition |
| Failed or recovering | Keep explicit failure; preserve a usable source if possible, fence motion, retain evidence and never silently re-arm real hardware |

A distributed commit cannot be assumed simultaneous. Until all required applied
receipts agree, consumers refuse cross-generation observations/intents. A
partition after one participant applies leaves motion inhibited; it is not
reported as a successful atomic switch. Persist transition intent/result for
reconciliation, but never persist a real actuation permit across process/host
restart. A prepared destination must not be allowed to publish active commands.

Use separate generation domains:

- Source/scene reset: new logical source session; clear target/range/ROI joins,
  selector and controller measurement histories. Reconcile SIM score/session
  policy explicitly. Do not reset the hardware's physical coordinate origin.
- Tracking mode: preserve source session, advance processing revision, invalidate
  previous selection/ROI work and reset target-dependent controller history.
  Preserve current explicit seed/release semantics and do not restart inference.
- Backend/profile/calibration: new backend/control epoch; fence old intents,
  pending serial motion and pose history. Never copy fake pose as real feedback.
- Score reset/recording toggle: independent operation/session acknowledgements;
  do not suppress active catalogs, source freshness or applied-control indication.

Real activation needs a separately granted bench/operational permit bound to
the current backend/profile and valid panel/timing/feedback state. Ordinary
source commands, reconnection or an unchanged switch HIGH cannot create it.
Existing permitted real sessions may retain their permit through a confirmed
source transition only if scope remains valid; outputs still hold until new
source/pose validity is established. Tracking mode alone does not add authority.

Real-to-fake/disabled handover first requests stop and cancels pending motion.
An enqueue ACK is not proof of stop. On failed communication, accept only a
previously qualified firmware timed-command expiry bound; otherwise remain in
fault/unknown-real-state and require bench recovery. Do not report a successful
fake switch while physical command termination is unproven.

Alignment is explicit: measured real pose/step anchor, home reference, axis
signs/gearing and camera-to-mount calibration must agree. Software SIM may reset
its own origin under a new epoch. Physical zeroing/homing is a separately
authorized operation, never an automatic side effect of selecting Camera/HIL.
If alignment cannot be established without movement, defer real activation.

## Timing and data contracts

Reuse PerceptionSnapshotV2, ControlObservation/Intent/Diagnostics and existing
ROI/recording contracts where semantics remain valid. Their strict parsers
must not silently accept invented fields. For backend/generation additions,
introduce an explicit versioned contract/envelope and a tested compatibility
adapter; do not overload source names, coordinates or timestamp fields.

Keep capture identity, receive time, observation completion and actuator sample
time distinct. All feedback includes actual backend and sample origin. Repeated
publication cannot rejuvenate an old sample. Same-host Jetson camera control
uses its qualified sensor clock; PC SIM still requires PC-to-Jetson clock
mapping even after the controller moves to Jetson.

The PC renderer also needs a qualified mapping from render time to Jetson pose
time. Reuse the measured pose timeline/clock exchange where valid, recording
mapping uncertainty and interpolation delay. Do not extrapolate physical pose
silently or declare LAN latency negligible. Moving fake control to Jetson adds
a pose network leg; measure it before accepting tracking parity.

Retain the existing50Hz controller, nominal60fps input and720p30 return targets
for comparison; do not increase these rates while changing architecture.
Preserve actual configured geometry/bitrate, fixed16-frame ROI pending bound
and34-image clone-pool capacity unless separately reviewed. Fake plant integration is deterministic and has
its own bounded timestep independent of display redraw frequency.

Maintain capture-age/clock-uncertainty watchdogs and verify time-daemon drift
assumptions on current hosts rather than promoting the old observed survey into
a guarantee. PC/Pi latest HUD telemetry remains display-only. Optical exposure-
to-display timing is a separate measurement, not supplied by source identity.

## Console and feature placement

- Targets: Tracking mode YOLO/ROI. YOLO retains Planner selection and Eligible
  classes; joystick picker changes target. ROI retains Select ROI, Release ROI
  and Box size. No list of internal IDs or duplicate second lock box.
- Operations: Source and scene; explicit motion/execution preset and applied
  inhibition reason; Recording; conditional confirmed Clear Swarm score;
  receiver Reconnect video only when needed. Real choices remain unavailable
  while the hardware gate is closed. Backend status cannot be confused with
  a switch that grants automatic or manual permission.
- Information: wide scrollable System, Connections, Control, Panel, About;
  add requested/applied transition, backend/pose provenance, selected profile,
  hardware-vs-fake availability and useful timing/rate/queue metrics here.
- Settings: preserve Speed limits, Lost target, Detection persistence,
  Engagement rules and Calibration with current editing/confirmation guards.
  Clearly label which SIM/hardware profile is being edited; no shared-gain bleed.
- Mandatory HUD: source/tracking/authority/REC, E-stop/fire, freshness/inhibition,
  existing score/raw-error strip and receiver low-FPS/loss alerts. Keep the
  visibility-first layout, independent shared theme and no MPC term display.

PC remains a read-only viewer in the operational console design. Pi physical
panel/joystick is the operator interface; terminal/PC keyboard interaction is
not required to complete normal operations. Test-only input injection stays
private and must not publish synthetic authority onto live panel endpoints.

With a real panel configured, absent/stale/unreadable input inhibits control.
An explicit panel-less software test permit can supply synthetic panel state
only while backend=simulated, never real, and loss of an already present real
panel cannot activate that fallback. Preserve lamp off-delay versus authority
watchdog separation. Panel source/process reboot must recover through new
sessions without requiring stale frontier comparisons across clock epochs.

## Configuration and network changes

Replace `operator.modes.*` lists of service stop/start actions with validated
runtime presets and a capability matrix. Retain systemctl actions only in a
separate startup/recovery/deployment layer. Split camera input geometry/timing,
controller policy and backend-specific settings instead of loading one overlay
that mutates unrelated dimensions. Remove the old PC-only controller stacks
and truth-fed HIL defaults from operational presets after parity.

Keep strict duplicate/unknown-field rejection, resolved digests and reviewed
YAML/native-profile differences. Runtime state is an applied transaction, not
another host overlay. Live/idle settings retain safe-point validation; static
driver/model/port changes remain explicitly cold/restart-required maintenance
unless a separately tested hot path exists. Do not convert every restart-tier
setting into an untested runtime mutation.

The operator agent serializes structural config and runtime transitions but
does not use one broad busy flag to withdraw otherwise valid target/activity
telemetry for score reset or recording. Coordinate shared values only with
their actual required consumers. A disconnected optional PC must not block a
same-host Camera/Pi operation; SIM input and PC recording can be unavailable.

Preserve PC-initiated control/config connections so the Jetson does not require
a fixed PC identity or inbound PC command endpoint. Renderer, clock and return
destinations need explicit session-scoped endpoint registration/validation;
the existing fixed return_ip/PC clock address cannot be called identity-free.
Keep Pi's independent return route. Retired PC sessions cannot steal an active
route or renew source freshness. Apply destination changes without resetting
the shared encoder/other receiver; qualify RTCP receiver routing separately.

Retain explicitly named publisher owners for target selection, pose, applied
activity and recording state. The new controller/bridge path must not leave
the old PC publisher bound to the same operational endpoint. Any compatibility
relay is read-only, epoch-tagged, temporary and inventoried in deployment.

Current operator channels assume a trusted LAN; UUIDs, revisions and epochs are
not authentication. Keep existing privilege/allowlist boundaries, scoped local
IPC for hardware writes and separately granted real permission. Do not expose
new unauthenticated Internet-facing real-activation endpoints. Any deployment
beyond the trusted LAN needs an explicit authentication/transport review.

## Lifecycle and failures

- At boot, UI/agents/panel can run with missing camera/PC/motors. Desired source
  may restore a configured software/preview preset, never real permission.
- Standby parks/closes optional ingest/render work through the runtime while
  preserving status/menu and recording control. It does not arm or silently
  start a bus owner. Warm-resource idle policy is measured and explicit.
- Perception/renderer failure invalidates observations; controller/bridge
  watchdogs withdraw motion. A replacement process has a new session and cannot
  accept old locks/permits. Receiver reconnect never reseeds ROI.
- Bridge failure leaves real recovery dependent on qualified command expiry,
  not an assumed graceful stop. Serial restart does not automatically enable
  motors, replay queued commands, rehome or restore real authority.
- Coordinator failure invalidates its transition/permit lease; participants
  reconcile actual state on return rather than replaying saved real activation.
- An optional viewer/HUD relay/recorder failure does not restart perception,
  controller, panel or renderer. Report storage/video-recording faults without
  disguising them as successful recording or stopping control implicitly.
- Preserve native worker join/EGL lifetime rules. No teardown fault is concealed
  with `_Exit`, unbounded ignored worker ownership or a second GPU owner.
- Shutdown stops/fences the controller, performs bridge stop/release while
  authorized serial IPC remains alive, then closes serial. Record timeouts and
  unknown physical state; never claim serial shutdown itself stops motors.

## File and module implementation map

Proposed new module names are planning targets, not claims that files exist.

| Boundary | Existing code to reuse/change | Planned extraction or addition |
|---|---|---|
| Runtime protocol | `common/operator_commands.py`, `common/pc_mode.py`, `common/visual_roi_seed.py`, `common/schemas.py` | `common/runtime_state.py`: typed state, generations, receipts, compatibility and capability validation |
| Transition owner | `jetson/operator_agent.py`, `jetson/pc_mode_channel.py`, `pc/mode_agent.py` | `jetson/runtime_coordinator.py`: prepare/commit/reconcile state machine; replace service-mode tables, retain outbound PC connection |
| Motion backends | `jetson/gimbal_bridge.py`, `common/gimbal/gray_box.py`, `common/gimbal/mks_servo42_rs485.py` | `jetson/motion/backends.py`, `simulated.py`, `mks.py`; extract current bridge phases without weakening gates |
| Serial fence | `common/serial_io.py`, `tools/serial_io_service.py` | Backend epoch/permit validation, queue fence/terminal receipts and isolated fake serial fixtures |
| Controller | `jetson/control/video_runtime.py`, runtime config, observation assembler, PID/estimator/timing modules | Resident session/profile adapter; eliminate PC operational controller ownership and truth endpoints |
| Input handover | `jetson/deepstream/runtime.py`, `pipeline.py`, Argus sensor metadata and verified RTP join | `jetson/deepstream/source_switch.py`: source adapters, canonical geometry, reset/warmup and epoch-specific publication |
| Resident tracking | `resident_roi.py`, `roi_runtime.py`, `roi_pipeline.py`, `jetson/visual_lock/*` | Keep current resident path; integrate runtime generation/capability receipts, no replacement tracker rewrite |
| Rendering and pose | `pc/streamer.py`, `pc/clock_sync_service.py`, scene/score/ROI helpers | `pc/sim_runtime.py`: runtime scene/pose commands; remove operational local plant/intent consumption and retain diagnostic tools explicitly |
| Presentation | `pc/ui.py`, V2 HUD, `pc/hud_relay.py`, `rpi/operator_display.py`, `rpi/display/*` | Source-independent subscriptions/state, new menu/runtime receipts, generation-aware recovery; keep shared theme and drawing ownership |
| Panel | `rpi/runtime_control.py`, `rpi/panel_indicators.py`, `tools/sim_panel.py` | Reuse authority policy as a library in Jetson controller; private fake-only test publisher, no production PC panel process |
| Recording | `tools/flight_recorder.py`, `rpi/display/recording.py`, SIM score events | Resident recording session controls, cross-host receipts, transition metadata and quota/finalization tests |
| Config | `configs/base/*`, `configs/controller/*`, `configs/sim/*`, schema/fields/live/shared config | Independent profiles/presets, compatibility rejection and granular hot/cold application receipts |
| Services/deploy | `deploy/systemd/{pc,jetson,rpi}`, `tools/deployment.py`, `scripts/deploy.sh`, process guards | Versioned26-to13 inventory migration, new startup groups, disabled real startup, rollback-aware verification |
| Evidence/docs | Existing unit/native/transport/UI/control tests and verification tools | Contract/transition matrix, benchmark comparison report, current architecture/operator/launch/config docs and journal |

## Ordered work packages and exit gates

### P0 Baseline and contracts

Freeze deploy-111 unit/config/model/native digests, current service states and
the resident ROI live evidence. Inventory every menu/control/recording feature
using the preservation table. Add runtime-state/backend contracts and pure
state-machine tests before replacing deployed command paths. Define per-owner
readiness, bounded operations, shutdown and resource budgets.

Exit: parser/state/replay tests cover all valid/invalid combinations, retries,
restarts, partitions and authority fences. Baseline scripts/reports are retained
outside source. No runtime deployment or physical owner starts.

### P1 Fake-capable bridge and independent SIM profile

Extract the shared intent gate and real adapter behind a fakeable interface.
Build deterministic simulated backend, applied-state provenance and separate
SIM profile. Isolated software tests must prove no serial/device imports/open,
port writes, parameter setup or hardware readiness claims in fake mode. Add
serial queue fencing tests using a protocol fake, not a real tty.

Exit: existing bridge/serial safety contracts plus fake-clock motion and switch
tests pass; real tuning changes do not alter the stable-SIM response. Production
hardware units remain inactive. Do not retune hardware while extracting code.

### P2 One Jetson controller and pose-driven PC renderer

Move the software controller input/output path to the Jetson fake bridge while
retaining image-derived V2. Add tagged pose subscription/clock mapping to PC,
remove double integration and adapt Single/Swarm score/raw-error/HUD. Port real
panel authority directly to the Jetson with explicit fake-only test isolation.

Exit: bounded deterministic texture -> Jetson visual tracker/controller/fake
bridge -> PC pose-driven render loop closes; PID and FF components remain
separate and replayable. No truth-to-controller/tracker input, unauthorized
commands or simulated feedback on physical availability. Measure added pose
network delay and tracking parity; retain the old rollback release if it fails.

### P3 Persistent PC renderer viewer and recording

Add frame-boundary Single/Swarm changes, reusable scene assets, runtime source
pose changes, unified read-only viewer and resident recorder controls. Source-
or viewer-lifetime changes no longer restart panel/controller/recorders.

Exit: same PC PIDs across repeated scene/view changes, correctly scoped score
and lock reset, recording segmentation and quota behavior, private panel menu
tests. Viewer/recorder failure cannot cascade into renderer/control shutdown.

### P4 Persistent perception source switch

Implement/measure warm camera and RTP ingest behind common perception/return.
Qualify canonical geometry, source clocks, NvDCF reset, ROI source invalidation,
return mapping, output RTP and native memory ownership. Keep existing YOLO/ROI
commands and worker behavior. Stop at a measured capacity/driver limitation;
do not label a cold pipeline recreation as seamless.

Exit: Camera/SIM source switches preserve shared process/encoder/receivers,
fresh generation-specific metadata, ROI correctness and measured video budgets.
Run control-free first, then software motion. No physical trial is required.

### P5 Runtime coordinator console config and telemetry

Replace ordinary stop/start modes with typed presets/transitions. Implement
requested-versus-applied status, PC identity/endpoint reconciliation, panel/menu
integration, score/recording independent operations, shared config exclusion
and boot/standby/failure behavior. Update lamp input provenance to new bridge
receipts without changing the electrical policy.

Exit: deterministic menu and private panel tests exercise every software preset,
mode, tracking workflow and injected failure, including no-panel behavior.
Camera/Pi mode works with optional PC offline; SIM fails closed without PC.
Real transitions can be tested with a quarantined protocol fake but remain
unavailable on production hardware until separately authorized.

### P6 Managed rollout and removal

Extend deployment's allowed-add/remove inventory explicitly; its present refusal
of removed units is a safety gate, not something to bypass. Stage clean tagged
releases, review all config/unit differences and preserve model/native hashes.
Archive old unit fragments and restore maps before removing references. Migrate
one host at a time in a software-only maintenance window; compatibility permits
old/new control-plane readers without simultaneous old/new GPU/control owners.

Before activation under the existing bridge/controller service names, obtain
explicit approval for the updated fake-only startup policy and reconcile the
standing AGENTS.md activation restrictions. A backend flag or service rename
does not itself lift that restriction. Until then, the extracted fake runtime
can be tested in bounded isolated software processes; physical activation
still requires the independent wiring/RS485/bench gate.

Exit: managed validation/activation/verification/audit on all hosts,13 service
roles with no unmanaged overrides, active source proven from /proc and manifests,
boot policy checked with real permission absent, and rollback rehearsed in
software. No GitHub push, commit or deployment is implied by authoring this plan.

Only after parity remove obsolete installed services/targets and production
imports. Preserve useful standalone tools, reports, recordings, model stores,
native sources and rollback releases. The82 failed transient test-unit records
from the current inventory are bookkeeping, not services to consolidate; archive
their evidence and clear exact stale records separately when authorized.

### P7 Physical bench qualification

Deferred behind AGENTS.md: repaired wiring, required-axis responses and explicit
user authorization. First read-only real feedback, stop/expiry/fence tests and
coordinate alignment; then bounded unloaded real-backend handover and Camera/
HIL trials with stated envelope and stop protection. Test real-to-fake loss and
reconnection without implicit re-arm. Loaded operation and actual camera tracking
quality are subsequent gates, not proven by successful software rollout.

## Acceptance matrix and measurements

Pure unit/contract tests precede GPU/live tests. Use deterministic detections
for selection/control tests and textured native pixel fixtures for ROI; reserve
learned-model effectiveness for frozen labelled data. All tests use bounded
durations, exclusive owners, retained failures and restoration receipts.

| Boundary | Required acceptance |
|---|---|
| Modes | Every supported preset and disallowed combination; both directions of each source/backend/scene transition; duplicate/out-of-order/CAS conflict and timeout cases |
| Ownership | Exactly one perception/encoder, controller, bridge and selected pose integrator; one physical bus writer only when explicitly authorized; no old PC controller |
| Resident continuity | Ordinary source/scene/tracking changes retain relevant service PIDs; processing mode retains source session; source changes advance logical identity while output remains stable |
| Switching/video | For each qualified warm software switch: no process/decoder-bin rebuild attributable to that switch, no black/frozen substitute counted as live, maximum decoded gap <=100ms and return >=28fps under the declared loss-free lab fixture |
| Sustained performance | Nominal60fps input/50Hz control/30fps return; same-host before/after comparisons on matched scenes with no-load/warm/ROI loads; record actual rates, misses, queue/drop/lag p50/p95/p99/max, CPU/GPU/memory and thermal state |
| Control and timing | All source/pose maps qualified; zero cross-epoch or unauthorized nonzero outputs; exact capture age retained; feedback/FF traces and fake plant response match replay. Reject any statistically material tracking/latency regression until explained, not traded for service count |
| Physical fencing | Fake-peer tests prove no old queued nonzero command reaches the new backend, enqueue ACK cannot qualify stop, permit loss inhibits control and restart cannot restore real permission; real proof deferred to bench |
| Panel and ROI | Manual/auto/E-stop/unreadable/high-at-start/off-on edges, lease loss, Manual OFF lock retention, repeated lock/release, poor-texture refusal, source change during acquisition, worker failure and late result/seed rejection |
| Targets and depth | Planner on/off, nearest lock/reacquire, class eligibility, learned/rule policy parity, internal IDs/rank badge, partial-box held-range expiry, unknown ROI depth and no stale parallax |
| Presentation | PC/Pi feature inventory, no duplicate boxes/facts, mandatory overlay foreground, smooth correctly projected tapes, mode colors, score/error strip and explicit latest-telemetry scope |
| Recovery | Drop/delay/reorder/corrupt video, clock jump/map expiry, no/late pose, native worker failure, process crash, lost PC/Pi, disk full and mid-commit partition; video may report unavailable, motion must remain fenced |
| Recording/config | On/off/failed writer receipts, decodable keyframe-aligned segments, transition/source metadata, shared/local keys and idle guards, config-transition exclusion; score reset does not suppress catalogs/lamps/lock |
| Lifecycle | Startup without hardware/PC/panel, source process restart, safe standby, orderly stop and abrupt owner loss; source epoch recovery without cross-clock stale-frontier lockout |
| Endurance | After bounded functional gates,30-minute repeated software transition run with at least100 tracking flips,20 source switches and20 scene changes; count failures, memory trends and resource ownership, not just final health |

The100ms/28fps values above are proposed warm-switch lab acceptance thresholds,
not production latency guarantees. Deploy-111 menu evidence is approximately
29.97fps/max57.95ms; moving native/software evidence includes worker lag p95
85.94ms. Establish a repeated baseline before selecting percentile regression
budgets; never omit that backlog or claim hard real-time behavior from these
samples. Cold readiness gets its own measured/reported budget and failure path.

Each receipt records code/dirty state, config/profile/model/native/plant digests,
host/runtime/clock assumptions, scenario/seed, packet/frame identity, owner PIDs,
transition/epochs, raw metrics, acceptance and restoration paths. Test artifacts
remain outside source. Journal entries distinguish planning, implementation,
unit/native proof, software deployment and physical qualification.

## Rollback and completion

Runtime rollback may restore a previous software source after verified owner
receipts; it does not replay previous physical permission or old targets. An
unconfirmed real stop stays faulted. Release rollback restores archived unit
fragments/config/native/model references in documented order, with no physical
autostart and no overlapping bus/GPU owners. Retain deploy-111 as baseline.

Completion means all26 service mappings and feature rows have implementation
and evidence receipts, all software gates pass on installed immutable sources,
normal console transitions no longer dispatch service start/stop, and current
architecture/launch/menu/config/deployment documentation agrees with manifests.
Physical readiness remains explicitly pending until P7 passes; a complete
software migration must not be labelled accepted physical tracking.

Primary current references: [deployment](deployment.md),
[controller](controller_architecture.md), [resident ROI](visual_lock.md),
[panel/menu](operator_display.md), [configuration](configuration_architecture.md),
[serial lifecycle](serial_io_service_definition.md) and
[verification](verification_strategy.md). Older Single truth-control and
standalone ROI menu passages describe earlier implementations; deploy-111 code
and its newest receipts take precedence until those guides are reconciled.

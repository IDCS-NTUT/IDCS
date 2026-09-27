# Video controller: measured timing, raw PID, then target-motion feedforward

(Developed as "controller V3" on branch `v3-pid-hardware-verification`,
merged into `main` on 2026-09-27; the V2 controller it replaced was removed.)

Status: **bounded unloaded yaw-plus-pitch-A PID/FF hardware trials and matched live simulator-video HIL verification complete; production real-camera timing and net FF efficacy remain unqualified**.
The V3 video controller has an opt-in guarded command path. Matched
feedforward-off/on and within-run crossover trials have exercised it with
real yaw and pitch-A motors, simulator frames, Pi safety, and a test-only
clock policy. They validate the architecture and measured timing, but do
not show consistent net video tracking benefit from FF at the current
integer-RPM actuator resolution. The separate Jetson-local synthetic-target
hardware trials showed benefit on controlled sine trajectories; neither
result qualifies real-camera timing.
This controller is deployed as `idcs-controller.service` (sim-camera HIL
overlay). The V2 controller (`jetson.control_runtime`,
`qualified_controller_profile`, `shadow_rate_policy`, the LOS Kalman study
environment) was removed on 2026-09-27; see the migration journal.

## Audit of the path before this controller

The V2 controller loaded an offline simulation report and drove one
monolithic policy that formed a Kalman-predicted position error, gimbal-rate
damping, and target-rate feedforward in a single decision. Disabling its
feedforward gain did **not** produce a raw PID baseline, and it had no
integral term.

`pc.streamer` sends a source timestamp in a ZMQ header separate from the RTP
video frame. `jetson.deepstream.header_correlation` associates the next
header with the next decoded frame by order, not an identity carried inside
the video. A dropped frame on either path can silently assign the wrong
source time. DeepStream also rounds its source/receive/inference timestamps
to milliseconds before publishing V2 perception. Neither property meets a
causal feedforward timing contract. Until in-band frame identity is verified,
the new timing gate rejects that association.
The PC timestamp is taken at completed frame retrieval (or simulator render),
not at physical sensor exposure; exposure-to-retrieval delay remains an
unmeasured component for real cameras.
For the Jetson IMX219 1280x720@60 Argus path, a clean 60-second source-only
survey measured first-sensor-data-arrival to `nvarguscamerasrc` source pad at
6.822 ms median and 7.530 ms p99, with no frame-number gaps. A separate
DeepStream run measured source-pad to inference input at 0.166 ms median and
inference/metadata at 10.750 ms median. These stage percentiles must not be
summed into an end-to-end percentile. The plugin exposes sensor frame number
and timestamp, but Argus-mode V2 perception does not yet carry them into a
verified V3 capture-age mapping. Optical exposure-to-sensor-data timing also
remains unmeasured; the survey does not authorize V3 motor control.

The PC `enp4s0` and Jetson `enP8p1s0` Ethernet interfaces advertise no
hardware PTP clock or hardware transmit/receive timestamps. Software
four-timestamp exchange gives an **offset interval**, not an exact offset;
its width includes network asymmetry. Clock drift between exchange and frame
also needs a measured or specified bound. The current one-second exchange
cadence and five-second mapping lifetime are therefore not proof of exact
frame age. Ethernet capacity is not the limiting factor: per-frame event
metadata can be sent without reducing video cadence, but a side channel
alone cannot prove which decoded frame owns it.
The opt-in frame-identity path now attaches frame ID and source monotonic
nanoseconds to GStreamer source buffers, reads those reference metadata on
the *encoded RTP marker packet*, and sends the resulting `(SSRC, RTP
timestamp) -> source event` mapping on a dedicated low-latency Ethernet
side channel. The Jetson reads the same key from the jitterbuffer marker
packet and joins it to DeepStream's decoded-frame PTS. The join is exact and
fail-closed; the old FIFO header path remains the default rollback. RTP
does not itself carry source time, so the side-channel header is still
required. Sender and receiver buffer/list behavior and H.264 loss were
tested, but a broader reorder/loss and clock-drift qualification remains.

## Separated contracts

1. **Frame identity and timing.** Capture, encoder/payload, Jetson receive,
   inference, snapshot publication, controller decision, serial acceptance,
   and encoder feedback are distinct events. Every timestamp names its clock
   domain and frame/command identity. Same-host differences are exact to the
   software timestamp point; cross-host differences are intervals with a
   recorded uncertainty and drift bound. Missing identity or bounds means no
   cross-host prediction. The first pure contract lives in
   `jetson/control/timing.py`.
2. **Basic feedback PID.** Use the measured bearing error, not a Kalman-
   predicted error. P and I act on error; D acts on measured gimbal rate to
   avoid derivative kick and keep target-velocity feedforward separate.
   Conditional integration, rate/acceleration limits, loss holds, and exact
   elapsed decision time are explicit in `jetson/control/pid.py`. The
   baseline has no estimator, network, or actuator access. The serial-free
   `jetson/control/shadow_pid.py` adapter accepts only schema-valid
   `ControlObservation` records with verified source identity, named clock
   domains, conservative age bounds, and fresh gimbal/safety data. It emits
   an immediately expired `mode=shadow` intent. Its synthetic replay is
   versioned under `tests/fixtures/control_pid_replay_v1.json` and is
   executable with `python -m tools.replay_control_pid ... --verify-golden`.
3. **Target-motion estimator.** After frame identity and timing are verified,
   fuse each distinct source-frame bearing with the gimbal pose at that
   frame's exposure/capture time. Expose innovation, rejection, covariance,
   rate, and age; do not silently substitute a receive time or current pose.
4. **Feedforward composition.** Add a separately bounded target angular-rate
   contribution *after* the raw PID baseline is accepted. Log raw PID,
   feedforward, pre-limit demand, and final command independently. Suppress
   feedforward when timing, frame identity, or estimator quality fails; keep
   raw feedback available only within its own freshness/safety contract.
5. **Actuator timing.** Measure controller decision to serial enqueue,
   acceptance, and encoder response per command token. Do not call serial
   acceptance motor application or predict an unmeasured future actuation
   time. Any lead compensation needs a measured delay distribution and an
   explicit uncertainty gate.

## Verification gates

The new pure timing/PID components have deterministic unit tests. The opt-in
identity canary published 181/181 decoded frames with verified source time
at roughly 30 fps, no ambiguous/withheld joins, and sub-millisecond source
timestamp precision; no motor authority was used. A deterministic 200-frame
loss/reorder join test and versioned eight-observation raw-PID golden replay
pass; neither qualifies a live timing bound or hardware gain. A synchronized,
isolated PC-to-Jetson canary then dropped four entire RTP frames and thirteen
independent headers among 180 generated frames. DeepStream decoded 175 frames,
withheld exactly thirteen for absent headers, and published 162 verified,
monotonic snapshots with zero ambiguous joins. H.264 recovery varied sharply
when startup or loss was more severe, so this is a bounded transport result,
not a packet-loss tolerance guarantee.

A separate 55-second, 50 Hz PC/Jetson software-clock survey returned 2,451
valid exchanges. The narrowest offset interval was 3.33 ms and the median
width was 3.71 ms; all intervals overlapped during that window. The valid
exchange span was 49 seconds, shorter than the requested 55 seconds. A
midpoint fit suggests +4.34 ppm relative drift, but the lower-latency half
suggests +0.87 ppm. Under a *constant-slope* model, all exchange intervals
admit roughly -72 to +73 ppm. The `tools.analyze_clock_drift` report retains
these distinct quantities and explicitly does not call any of them a future
guarantee. A longer, complete five-minute, 50 Hz survey collected 15,000
valid exchanges spanning 299.98 seconds. The all-sample midpoint trend was
+0.65 ppm and the lower-latency half gave +0.61 ppm. All 15,000 offset
intervals admit constant slopes between -10.91 and +12.10 ppm; this is an
observed-window interval *under a constant-slope model*, not a certified
future oscillator bound. The survey tool now reports actual valid span and
fails its completeness check when that span is below 95% of the requested
duration.

The `jetson.control.clock_watchdog` can reject stale exchanges and
contradictions against an externally justified limit, but defaults to no
qualified drift bound and therefore no controller clock mapping. Wide
software-timestamp intervals can hide small frequency changes, so passing
the watchdog is not proof of its configured limit. The V3 timing gate still
refuses to extrapolate a clock exchange without a measured or specified
drift bound. A first live-video, non-authoritative shadow run now verifies
this integration path: 300/300 decoded frames carried source identity, the
missing-policy branch held 300/300 decisions, and an explicitly empirical
20 ppm shadow-only policy produced 299 raw-PID tracking decisions with a
guaranteed moving synthetic target. It used synthetic gimbal and safety
inputs, so neither hardware gains nor real closed-loop behavior were tested.

The next shadow policy caps capture age at 80 ms, clock-sample age at 100 ms,
and the observed offset-interval width at 15 ms. A provisional 1000 ppm drift
stress assumption then gives a conservative maximum mapping-interval width
of 15.390392 ms over a 195 ms span, under a 20 ms shadow-study budget. The
watchdog validates this arithmetic at configuration time and latches a fault
on an over-wide exchange. The 15 ms threshold exceeds the 12.04 ms maximum
seen in a ten-minute active-video survey; it is an empirical margin, not a
network-service guarantee. The 80 ms cap is a study criterion, not a proven
real-target pointing tolerance; PC source time is still frame retrieval or
sim render rather than physical exposure. The 1000 ppm value is deliberately
much wider than the five-minute constant-slope observations, but is not a
measured future oscillator guarantee. These limits are not connected to V2
or motor authority.

The active-video ten-minute survey returned 29,905 valid exchanges across
598.08 seconds. Its maximum width was 12.04 ms, and a 3,000-sample slope
subset admitted constant PC-minus-Jetson drift between -5.47 and +5.89 ppm
for that observed window. The revised 15 ms gate exceeds the observed width
maximum. A repeat 300-frame verified-video shadow canary held one frame at
86.05 ms against the 80 ms capture-age limit, held one clock-warmup frame,
and tracked the other 298 under the provisional 1000 ppm assumption. The
missing-bound branch held all 300. This demonstrates the fail-closed shadow
path, not live oscillator qualification or hardware control efficacy.

Next, qualify an operational drift bound across representative load and
temperature changes, then run V3 in shadow beside V2 with real observations
and no command publication. Only then begin bounded unloaded hardware PID
validation. Kalman rate estimation and feedforward are subsequent gates,
with independent on/off/standalone attribution trials. The existing
`docs/verification_strategy.md` hardware safety requirements continue to
apply. Simulated camera dynamics never qualify real hardware gains.

### Isolated unloaded yaw PID evidence (2026-09-26)

The bounded 18-second `pose18` trial used a Jetson-local synthetic yaw reference
`0, +0.06, -0.06, 0` rad, `BasicPID` gains `(8, 0, 0.1)`, 0.2 rad/s command cap,
3.5 rad/s² slew limit, 0.15 rad yaw and 0.03 rad pitch travel guards, fresh
Pi manual-safety and encoder gates, 50-ms intent validity, 100-ms firmware-
timed F6 commands, and an exclusive serial owner. It used no detector, video,
PC timestamp, plant model, or prediction. The bridge's pitch command was zero.
The first run (`first18`) revealed that optional encoder-rate fields were
being treated as required, holding 667/900 ticks and causing no measured yaw
movement; this was a software defect and that run is rejected. After the
fresh-pose fix, `pose18` tracked on 899/900 ticks, moved from home
`-2.41410` rad through `-2.33664` and `-2.50077` rad, and returned to
`0.00422` rad from home. The serial service reported 0 write failures and
0 uncertain writes, with 412 coalesced/superseded commands from 3,045 admitted.

The settled positive, negative, and return windows had mean absolute yaw
errors about 0.0049, 0.0102, and 0.0043 rad respectively while nonzero PID
rates continued. The configured 1:1 gear ratio and integer-RPM F6 payload
quantize all requested rates below `2π/60 = 0.10472` rad/s to zero. Every
command in those windows fell below that threshold; measured span was at most
two encoder counts. This is a demonstrated firmware/protocol command-resolution
limit, not evidence of PID steady-state convergence to zero error. Missing
encoder-rate estimates still forced zero derivative on 632/900 ticks; the
hardware trial establishes a yaw feedback baseline, not fully qualified D
behavior or proof that bus scheduling is no longer limiting.

A matched 18-second P-only run (`p_only18`, `Kd=0`) yielded 899/900 valid
tracking ticks, 0.01825 rad whole-run RMS error, and a final offset of
`-0.00153` rad from home. Its settled positive/negative/return mean absolute
errors were 0.00256/0.00835/0.00164 rad: marginally better than the
intermittent-rate PD run. All settled commands were still below 1 RPM and
encoded as zero. The P-only run is the safer provisional hardware baseline;
no claim that the derivative term helps is supported.

The codebase already contains `SpeedCommandDither`, so integer-RPM resolution
alone is **not** an irreducible hardware bottleneck. A further matched P-only
trial (`dither_p18`) applied that helper before the bridge. It retained
891/900 valid tracking ticks and zero serial write failures, but mean settled
errors worsened to about 0.0177–0.0178 rad and measured yaw span in each
settled window increased to 0.066–0.069 rad. This adaptation is rejected:
20-ms alternating rate/zero intents, 100-ms firmware command duration, and
priority/coalescing behavior do not make a time-accurate low-rate actuator.
The present bottleneck remains partly software actuation scheduling. A
20-ms firmware-timer repeat (`dither20b_p18`) reduced the oscillation but
still had settled errors of 0.0138/0.0118/0.0110 rad and settled position
spans of 0.054/0.045/0.054 rad. Its serial feedback had zero write failures,
217 superseded and 45 preempted commands. The first attempt at this variant
never reached motor actuation because a one-field YAML override replaced the
whole `gimbal` map; the successful repeat used a digest-checked full override
with only the command timer changed to 20 ms. Both dither timings are rejected.
Do not enable this dither option for normal control or claim hardware-only
closure.

Evidence: `/home/idcs/idcs-devtools/evidence/v3_pid_{first18,pose18,p_only18,dither_p18,dither20b_p18}/` on
Jetson and mirrored trace/logs under `C:/Users/Lab412/idcs-dev/evidence/`.
The hardware override SHA-256 was
`8c77c528cff2c6a7ab48aea7c91da6ebcd0374e15e64eeb1f0b11d0d0ab0f6f1`;
the accepted trial source SHA-256 was
`0039c881857249092e4cfe0aff3c538b568beabde3fc32dbbc58d6e2314de44d`.
The accepted PD-trial code revision was `29aecabd55c32efc8e7ef95bb68642e704ce2a98`;
P-only comparison used `dcf3d593cde716bd2bdf75df633e91246d799603` and
the dither evaluation used `bdc5e036d137f3e55cfd394db950cadae814cd77`.
The 20-ms repeat used `f6823ebefc19ca18639974cae82ed2a43a8493f1` and
full timer override SHA-256
`a557ebfe216b113d855acd72eb284bb66f58e7afe49eb31ea97780cd5e32a8a5`.
The local and full isolated-host suites passed (411 and 432 tests respectively).

Remaining gates: design and verify execution-time-aware sub-RPM actuation
without the rejected dither limit cycle; replace the absolute pitch-A/B
comparison with an origin-aware, mirrored-sign relative-motion watchdog;
qualify a continuous, source-timestamped yaw-rate estimate for D feedback;
and complete the
real-video frame-identity/clock-bound qualification before allowing the V3
video runtime to publish live intents. No software- versus hardware-bottleneck
claim is made for those untested paths.

The separate V2 video/HIL controller completed a bounded two-axis run on
2026-09-26 with exact simulator target truth and measured encoder camera pose.
Both pitch motors moved in mirrored raw-count directions after a first-canary
pitch-B dropout and direct timed B-only recovery; see the migration journal.
This does not qualify V3 two-axis command authority or prove the B dropout is
permanently resolved.

### Isolated yaw-plus-pitch-A PID and feedforward evidence (2026-09-27)

Because B later dropped out intermittently, an isolated bridge mode omitted B
motion and independently verified that its raw count did not change. Bounded
V3 local-target step trials selected P-only Kp=8 for yaw and Kp=4 for pitch-A
as provisional gains (Ki=Kd=0, 0.2-rad/s caps); their whole-run RMS errors
were 0.01793 and 0.01801 rad. A distinct timestamped constant-velocity
Kalman filter then estimated target rate. Its contribution is logged separately
and added before the same PID rate/slew limiter. No target velocity is inferred
from a missing or stale sample.

Matched unloaded 20-second simultaneous two-axis sine trials with an explicit
Jetson-local 60-ms observation delay changed exact-target RMS from 0.01648 to
0.01306 rad on yaw and 0.02684 to 0.01721 rad on pitch-A when FF scale 0.5
was enabled. Both runs tracked 996/1000 ticks; B stayed fixed, and the on run
had zero failed or uncertain serial writes. These establish an isolated
hardware-control baseline, not transfer of the 0.5 gain to camera detections.
The same gains and FF scale improved a held-out four-second target period:
yaw RMS 0.01373 to 0.01095 rad and pitch-A RMS 0.02337 to 0.01681 rad
in a matched simultaneous hardware pair, with B stationary and no failed or
uncertain serial writes.
The V3 video input remains shadow-only until source-clock policy and
frame-time-aligned camera pose support causal target-world-rate estimation.

The first rendered-simulator V3 video shadow (2026-09-27) now passes exact
source-frame verification and measured Jetson receipt/observation stamping.
`CameraPoseHistory` aligns the whole mapped capture interval only when
bracketed; `VideoTargetRateEstimator` estimates target world-angle rate from
aligned pose plus bearing. This calculation is separate from `BasicPID`.
The shadow runner compares PID-only with PID plus explicit 0.5-scaled FF;
both intents expire immediately and are never published. In 370 received
frames, 369 PID decisions tracked, 271 FF estimates were ready, and 92 stale
estimates contributed exactly zero. Only 15 ready frames changed the final
command by >0.001 rad/s because a stationary synthetic camera drove most
commands to the 0.2-rad/s cap. This is a functional boundary test, not a
tracking-performance result. The tested 1000-ppm clock policy is explicitly
empirical and shadow-only; no live video authority follows from it. Real
encoder pose, live safety state, justified clock drift, and bounded matched
off/on HIL validation remain necessary.

The read-only measured-CamState shadow closed two wiring gaps: P-only PID
now accepts a fresh measured pose without a rate field (D still requires
measured rates), and the video Kalman's 150-ms maximum sample age matches
the video PID capture-age limit. With the bridge read-only and the host
simulator consuming its CamState, the final 15-second run accepted 396/396
verified frames, tracked on 393 PID decisions, and produced 389 ready FF
estimates. The baseline and combined shadow commands differed by >0.001
rad/s on 164 ready frames. All commands were immediately expired and none
were published to the bridge. The serial event sample contained only encoder
and F1 queries. This verifies measured-pose *wiring*, not moving-camera
closed-loop behavior. The simulator may render the bridge's predicted pose,
which can diverge transiently from its encoder pose; source-frame pose
alignment and a qualified clock bound remain live-authority gates.

The candidate `jetson.control.video_runtime` now runs a 50-Hz loop with
clock polling in a separate thread. It stamps verified exact-frame snapshots
on Jetson receipt, consumes measured encoder CamState and real Pi safety,
then sends the observation through `VideoControllerCore`: raw bearing P-only
feedback (Kp yaw 8, pitch-A 4), independent Kalman target-world-rate FF
(scale 0 or 0.5), one rate/slew limiter, and a 0.15-rad projected trial
travel envelope. The default output is immediately expired shadow; any
timing, target, safety, or travel hold yields zero live candidate rates.
Live publication is disabled absent explicit test-only clock and unloaded
hardware acknowledgements. The bridge retains its 100-ms F6 motor timer,
own intent watchdog, hard angle bounds, and separate B-motion guard.

For simulator HIL only, target-world-rate estimation uses fresh bridge
`render_pan/tilt`, matching the host's rendered pose; real-camera estimation
uses encoder pose. Both require capture-time pose bracketing, at most 100-ms
sample gaps and 20-ms mapped time-interval width. These checks do not prove
the host applied a published render pose at the exact capture instant;
moving-camera HIL must measure that residual. The 250-ms video capture-age
gate is a bounded shadow/test policy motivated by observed source age
(median 169, p99 214 ms in the initial fixed-rate run), not a hardware PID
tuning result or a production latency promise. The final guaranteed-target
fixture varies within roughly +/-0.08 rad yaw and ~0.06 rad/s estimated
target speed; a 15-second render-aligned shadow tracked 747/750 fixed-rate
ticks with 739 FF-ready and no motor publication. Matched live FF-off/on
video/HIL trials remain unverified.

The later live HIL studies supersede that preparation status. The host now
tags every guaranteed-truth simulator snapshot with the exact relative pose
used for rendering and the timestamp of the applied Jetson CamState. V3
requires both in live HIL and holds motor rate at zero when pose age exceeds
100 ms; real camera operation still uses measured encoder pose and does not
accept simulator-only metadata. The dirty Jetson candidate perception schema
was left untouched: the isolated V3 runtime removes HIL-only metadata before
strictly validating the deployed V2 snapshot. A hardware-moving diagnostic
reduced apparent target-world-rate error from ~0.027/0.024 to
~0.001/0.002 rad/s yaw/pitch after this source-frame pose correction.

Both matched 20-second off/on pairs and two balanced 30-second crossover
pairs passed two-axis motion, pitch-B exclusion, serial-write, clock,
capture-age, and intent-lease checks. The off/on pairs disagreed on
feedforward efficacy; balanced cornered-path crossover changed yaw/pitch
RMS by -0.84%/+0.55%, and a separate smooth-path crossover by
+1.15%/-5.09%. Thus FF is causally timed, independent of PID and actually
applied, but *not* consistently beneficial under the 0.2-rad/s trial cap.
F6 carries integer RPM with a 1:1 ratio, so current serial writes contain
only zero or one RPM and the ~0.01-rad/s FF contribution is below one speed
quantum. This is the observed actuator-interface resolution bottleneck;
the cap is a test safety setting, not the motor's physical maximum. A
separate actuator-resolution design is needed before promising sustained
video-FF improvement or production deployment.

The simulator's vertical FOV is now supplied explicitly for render-pose HIL,
with the horizontal FOV and focal lengths derived from active frame geometry
in memory; real-camera calibration is not rewritten. A stationary target
under +/-0.04-rad synthetic camera yaw exposed a prior real/sim FOV mismatch:
false yaw-rate estimate 0.080 rad/s median, 0.115 p95. With the simulator's
60-degree vertical FOV (fx 935.31 px), a repeated non-actuating 15-second
shadow measured 0.00188 rad/s median and 0.00352 p95, 750/750 controller
ticks, 741 FF-ready. This is a camera-motion cancellation check, not live
motor or moving-target efficacy evidence.

### F5 absolute-axis actuation probe (2026-09-27)

An opt-in `gimbal.actuation_mode: f5_position` (default `f6_speed`) turns
accepted rate intents into bounded F5 absolute-axis targets
(`jetson/f5_actuation.py`, `jetson/control/position_target.py`) to
avoid the integer-RPM F6 quantum. Single-axis yaw bench probes showed F5
moves at the expected speed, but its coordinates are offset from the 0x31
encoder reading by an amount that grew 5 -> 12 -> 19 counts across
disable/enable cycles, while the motor's own 0x39 angle error stayed near
zero. F4 relative motion landed within ~3 counts. The committed mode
assumes zero offset and is not hardware-qualified. The agreed direction is
a per-enable-session step frame with a 0x39 lost-step watchdog; see the
migration journal. This is parked behind closed-loop tracking work.


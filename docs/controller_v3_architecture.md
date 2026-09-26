# Controller V3: measured timing, raw PID, then target-motion feedforward

Status: **offline foundation, opt-in verified frame identity, live-video raw-PID shadow canary, and isolated unloaded yaw PID hardware trial**.
The V3 video controller still has no command authority. A separate, bounded
Jetson-local synthetic-target harness has exercised `BasicPID` against the
unloaded yaw motor; it does not qualify the video timing, pitch axis, or V3
runtime for deployment. The existing V2 runtime remains the deployed controller boundary until each gate
below passes. Its offline-qualified Kalman/PID report is not a live controller
qualification; V2 feedforward remains off by default.

## Audit of the present path

`jetson.control_runtime` still loads an offline simulation report through
`qualified_controller_profile`, then drives the monolithic
`shadow_rate_policy`. That policy forms a Kalman-predicted position error,
gimbal-rate damping, and target-rate feedforward in one decision. Disabling
the feedforward gain does **not** produce a raw PID baseline. There is no
integral term in that policy.

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
   `jetson/control_v3/timing.py`.
2. **Basic feedback PID.** Use the measured bearing error, not a Kalman-
   predicted error. P and I act on error; D acts on measured gimbal rate to
   avoid derivative kick and keep target-velocity feedforward separate.
   Conditional integration, rate/acceleration limits, loss holds, and exact
   elapsed decision time are explicit in `jetson/control_v3/pid.py`. The
   baseline has no estimator, network, or actuator access. The serial-free
   `jetson/control_v3/shadow_pid.py` adapter accepts only schema-valid
   `ControlObservation` records with verified source identity, named clock
   domains, conservative age bounds, and fresh gimbal/safety data. It emits
   an immediately expired `mode=shadow` intent. Its synthetic replay is
   versioned under `tests/fixtures/control_v3_pid_replay_v1.json` and is
   executable with `python -m tools.replay_control_v3_pid ... --verify-golden`.
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

The `jetson.control_v3.clock_watchdog` can reject stale exchanges and
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

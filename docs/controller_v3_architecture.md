# Controller V3: measured timing, raw PID, then target-motion feedforward

Status: **offline foundation, opt-in verified frame identity, and deterministic raw-PID shadow replay**.
No V3 module has command authority. The existing V2 runtime remains the deployed controller boundary until each gate
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
Next, qualify an operational drift bound across representative load and
temperature changes, then run V3 in shadow beside V2 with real observations
and no command publication. Only then begin bounded unloaded hardware PID
validation. Kalman rate estimation and feedforward are subsequent gates,
with independent on/off/standalone attribution trials. The existing
`docs/verification_strategy.md` hardware safety requirements continue to
apply. Simulated camera dynamics never qualify real hardware gains.

# Launch procedure

How the IDCS services are started, stopped, checked and deployed. Every
service runs under systemd; the unit files are in `deploy/systemd/<host>/`
and each runs the code in `~/Desktop/project/IDCS-runtime` at a `deploy-N`
tag. Which configuration files each service loads is in `configs/README.md`.

## Hosts and units

| Host | Units | Kind |
|---|---|---|
| Jetson 192.168.0.5 | `idcs-deepstream-video` (PC video), `idcs-deepstream-camera` (own IMX219), `idcs-serial`, `idcs-bridge`, `idcs-controller`, `idcs-controller-camera`, `idcs-recorder`; targets `idcs-hil.target`, `idcs-camera.target` | system: `sudo systemctl ...` |
| PC 192.168.0.1 | `idcs-sim-streamer`, `idcs-sim-swarm-streamer`, `idcs-sim-controller`, `idcs-sim-swarm-controller`, `idcs-sim-panel`, `idcs-sim-ui`, `idcs-sim-recorder`, `idcs-hil-streamer`, `idcs-ui`; targets `idcs-sim.target`, `idcs-sim-swarm.target` | user: `systemctl --user ...` |
| Pi 192.168.0.3 | `idcs-manual` (safety panel: arm switch, E-stop, joystick, lights) | user: `systemctl --user ...` |

At boot: the Jetson starts `idcs-deepstream-video` (waits for PC video) and
the Pi starts `idcs-manual`. The motor stack is never enabled at boot, so the
gimbal never energizes unattended. Nothing starts on the PC.

Conflicting units are declared (`Conflicts=`): starting one mode stops the
other (for example `idcs-sim-swarm.target` stops `idcs-sim.target`,
`idcs-camera.target` stops `idcs-deepstream-video` and `idcs-controller`).

## Modes

### Simulation (no hardware)

```bash
systemctl --user start idcs-sim.target          # PC: fast-target scene, simulated mount
systemctl --user start idcs-sim-swarm.target    # PC: swarm scene, controller on DeepStream detections
systemctl --user stop idcs-sim.target           # (or idcs-sim-swarm.target)
```

Needs `idcs-deepstream-video` on the Jetson (running from boot). The sim
panel arms the simulated controller; the UI shows the return video.

### Hardware in the loop (real motors, simulated camera)

```bash
# Jetson, stack stopped: home both axes to their envelope centres
cd ~/Desktop/project/IDCS-runtime
PYTHONPATH=$PWD ~/Desktop/project/bin/python jetson/tools/home_axes.py \
    --config configs/base --config-extra configs/bench/uncoupled.yaml,configs/bench/tuned.yaml --execute
sudo systemctl start idcs-hil.target            # serial, bridge, controller, recorder

# PC
systemctl --user start idcs-hil-streamer idcs-ui
```

The controller moves the motors only while the Pi panel is armed (no manual
control, no E-stop). It steers on simulator truth by default; for the real
pipeline (YOLO + NvDCF) run it with `configs/controller/detections.yaml`
added (see Variants).

Stop and let the motors rest:

```bash
systemctl --user stop idcs-hil-streamer idcs-ui      # PC
sudo systemctl stop idcs-hil.target                  # Jetson
PYTHONPATH=$PWD ~/Desktop/project/bin/python -m tools.motors_off   # Jetson: F3 0, checks every ACK
```

### Local camera (the Jetson's IMX219)

```bash
sudo systemctl start idcs-camera.target        # Jetson: DeepStream on the camera, shadow controller, recorder
systemctl --user start idcs-ui                 # PC: return video
sudo systemctl stop idcs-camera.target && sudo systemctl start idcs-deepstream-video   # back to PC video
```

The controller runs in shadow mode (computes and records, no motor
authority): on the uncoupled bench the camera is not carried by the gimbal.

## Checking

```bash
systemctl --user status idcs-sim-streamer            # PC unit state (Jetson: sudo systemctl status ...)
journalctl --user -u idcs-sim-streamer -f            # PC log (Jetson: sudo journalctl -u idcs-bridge -f)
cat /run/idcs/deepstream-video-health.json           # Jetson DeepStream: frames, fps, return fps
cat /run/idcs-controller/controller-health.json      # Jetson HIL controller: reasons per tick
cat /run/user/1000/idcs-sim/controller-health.json   # PC sim controller
python -m tools.flight_log summary ~/idcs-flight     # recordings (Jetson: /var/log/idcs/flight)
```

Every service validates its configuration at start (`ExecStartPre ...
--check`); a bad config fails there with the reason in the log. To see
exactly what a service runs: `systemctl cat <unit>`.

## Variants and experiments

One-off variations run as transient units, which get logs, status and stop
like installed ones but disappear when stopped. Example: the HIL controller
on real detections.

```bash
sudo systemctl stop idcs-controller
sudo systemd-run --unit idcs-controller-det -p User=idcs \
    -p RuntimeDirectory=idcs-controller-det -p WorkingDirectory=$PWD \
    -p "Environment=PYTHONPATH=$PWD PYTHONUNBUFFERED=1" \
    ~/Desktop/project/bin/python -m jetson.control.video_runtime --config configs/base \
    --config-extra configs/bench/uncoupled.yaml,configs/controller/hil.yaml,configs/bench/tuned.yaml,configs/controller/detections.yaml \
    --health-file /run/idcs-controller-det/health.json
sudo systemctl stop idcs-controller-det
```

A variant used routinely should become a unit file in `deploy/systemd/`.

## Deploying a change

```bash
git tag deploy-N                        # on main, in IDCS
scripts/deploy.sh deploy-N              # all hosts; or: scripts/deploy.sh deploy-N pc,jetson
```

The script bundles the repository, checks out the tag in each host's
`IDCS-runtime` (refusing if that checkout has local changes), installs the
host's unit files and reloads systemd. It restarts nothing: restart the
services the change affects. `git describe --tags` in `IDCS-runtime` shows
what a host runs. A fresh Jetson checkout also needs
`scripts/prepare_jetson_runtime.sh` (DeepStream parser, models, policy engine).

## Troubleshooting

- **A unit fails at start**: `journalctl -u <unit>` (add `--user` on the PC
  and Pi). The `ExecStartPre ... --check` line names the configuration
  problem; `tools/runtime_process_guard.py` refuses when another process
  already runs the same module.
- **Local camera: Argus ends the stream at once** ("pipeline completed
  without DeepStream frame metadata" right after start): an Argus client that
  was killed can leave the daemon unusable. `sudo systemctl restart
  nvargus-daemon`, then start `idcs-camera.target` again. A unit using the
  camera must not set `PrivateTmp` (the client reaches the daemon through
  `/tmp/argus_socket`).
- **HIL streamer refuses to start** ("... of frames would have no measured
  pose"): the gimbal pose feedback is too sparse or late for the truth
  latency budget; check that `idcs-bridge` is running and publishing.
- **Motors stay energized after a stop**: stopping the stack sends zero
  rates but the motors hold position; run `tools.motors_off`.
- **Which code is running**: `git describe --tags` in `IDCS-runtime` on each
  host; `scripts/deploy.sh` keeps all three on one tag.

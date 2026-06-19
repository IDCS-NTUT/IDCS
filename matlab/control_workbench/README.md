# IDCS Control Workbench

This folder is a MATLAB/Simulink starting point for studying the IDCS control
system without video, ZMQ, YOLO, or hardware drivers.

The first target is intentionally small:

1. Load a recorded `control_trace_*.jsonl`.
2. Extract yaw/pitch command and CamState signals.
3. Fit the simple plant used by the repo's MPC model:

   ```text
   theta_dot = omega
   omega_dot = a_u * u - a_f * omega
   ```

4. Build a simple Simulink plant model for yaw/pitch.
5. Use MATLAB/Simulink toolboxes for tuning around that model.

## Quick Start

From MATLAB, run:

```matlab
cd("C:\Users\Lab412\Desktop\repo")
setupControlWorkbench

tracePath = "C:\Users\Lab412\Desktop\control_trace_1780802043.jsonl";
results = runTracePlantFit(tracePath);
sweep = sweepPlantDelay(tracePath);

buildGimbalPlantModel(results.plant);
open_system("idcs_gimbal_plant")

yawSweep = runPidSweep(results.plant.yaw, Axis="yaw");
```

The `results` struct contains:

- `results.trace`: raw decoded trace records.
- `results.yaw`: yaw command/CamState alignment.
- `results.pitch`: pitch command/CamState alignment.
- `results.plant.yaw`: fitted yaw `a_u` and `a_f`.
- `results.plant.pitch`: fitted pitch `a_u` and `a_f`.
- `sweep.best.delaySec`: the command delay with the best rate fit.
- `yawSweep.best.pid`: a first-pass PID candidate for the fitted yaw plant.

## Simulink Model Shape

The generated `idcs_gimbal_plant` model has:

- `yaw_cmd` input: yaw rate command intent.
- `pitch_cmd` input: pitch rate command intent.
- `yaw_theta`, `yaw_omega` outputs.
- `pitch_theta`, `pitch_omega` outputs.

This is only the plant shell. The next blocks to add are:

- PID Controller block or custom controller subsystem.
- Rate and acceleration saturation.
- Delay block.
- Reference/predictor subsystem.
- Stateflow target state logic for `track`, `hold`, and `lost`.

## Suggested Toolbox Use

- `Control System Toolbox`: response analysis and basic controller design.
- `Simulink Control Design`: linearize and tune Simulink control loops.
- `Simulink Design Optimization`: fit/tune parameters against traces.
- `System Identification Toolbox`: replace the simple least-squares fit with
  richer plant identification if needed.
- `Model Predictive Control Toolbox`: rebuild the repo's MPC in Simulink if you
  want a toolbox-native MPC path.

## Notes

The trace timestamps come from multiple processes. The workbench uses recorder
receive time to align `ControlCmd` and `CamState`, then applies a configurable
command delay. Start with `delaySec = 0.05` to `0.10` and adjust while comparing
the fitted response to the recorded CamState.

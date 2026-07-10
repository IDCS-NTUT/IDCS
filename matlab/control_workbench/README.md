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
mpcReport = analyzeMpcTrace(tracePath);

buildGimbalPlantModel(results.plant);
open_system("idcs_gimbal_plant")

yawSweep = runPidSweep(results.plant.yaw, Axis="yaw");
```

To build a camera-motion simulation from raw physical gimbal sweeps:

```matlab
sweepCsv = "C:\Users\Lab412\Desktop\repo\logs\gimbal_response_sweep_....csv";
gimbalModel = fitGimbalResponseModel(sweepCsv);
exportGimbalSimModel(gimbalModel, "configs\sim_gimbal_model.json");
```

To inspect raw sweep data without fitting a model:

```matlab
sweepCsv = "C:\Users\Lab412\Desktop\repo\logs\gimbal_response_sweep_....csv";
plotGimbalResponseSweep(sweepCsv, OutputDir="artifacts\gimbal_response_sweep")
```

The Python recorder also supports richer open-loop data profiles for later
offline analysis:

```bash
python -m jetson.tools.gimbal_response_sweep \
  --profile prbs \
  --axis yaw \
  --rates 0.1,0.5 \
  --profile-duration-s 30 \
  --seed 42 \
  --assume-exclusive

python -m jetson.tools.gimbal_response_sweep \
  --profile chirp \
  --axis pitch \
  --rates 0.8 \
  --chirp-start-hz 0.05 \
  --chirp-end-hz 1.5 \
  --profile-duration-s 45 \
  --assume-exclusive
```

The `results` struct contains:

- `results.trace`: raw decoded trace records.
- `results.yaw`: yaw command/CamState alignment.
- `results.pitch`: pitch command/CamState alignment.
- `results.plant.yaw`: fitted yaw `a_u` and `a_f`.
- `results.plant.pitch`: fitted pitch `a_u` and `a_f`.
- `sweep.best.delaySec`: the command delay with the best rate fit.
- `mpcReport.yaw` / `mpcReport.pitch`: MPC status, cost terms, references,
  predictions, solver iterations, and saturation extracted from `ControlCmd`.
- `gimbalModel.axes.yaw` / `gimbalModel.axes.pitch`: fitted open-loop camera
  motion parameters from the physical gimbal response sweep.
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
- MPC Controller block or custom QP subsystem, after trace diagnostics show
  which cost/reference pieces are worth porting first.

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

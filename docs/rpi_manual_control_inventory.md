# Raspberry Pi panel runtime inventory

Current managed entry point: `rpi.runtime_control`, launched by
`deploy/systemd/rpi/idcs-manual.service`. Reviewed 2026-10-11.

## Production responsibilities

- One session-locked owner polls the PCF8591 ADC and configured GPIO inputs.
- Joystick deadzone/mapping produces high-level rates in `ManualControlState`.
  The video controller, not the Pi, applies authority, travel and rate limits.
- The runtime publishes panel state locally for the display and manual state
  to the configured controller endpoint. It does not publish serial commands,
  open RS485, enable motors, home or zero axes.
- Indicator policy consumes fresh perception/applied-control/hardware feedback.
  GPIO12 gates auto only; GPIO5 manual selection is independent; GPIO16 gates
  fire-button indication only. E-stop dominates motion authority.
- GPIO12/16 are unreadable while their lamps are off; neither an old high nor
  an assumed off state may substitute for a fresh powered sample.

The complete switch/lamp state machine, freshness and hysteresis are in
[operator display](operator_display.md). Menu up/down inversion is only in
`PanelNavigator`; manual forward/backward pitch mapping is unchanged.

## Shared and retained legacy code

`rpi.manual_control` supplies shared GPIO/ADC helpers imported by the managed
runtime. Its standalone command loop still contains serial IPC initialization,
rate and stop/disable logic. It is **not** the managed Pi service and must not
be launched as a second controller. The extraction is therefore operationally
complete at the service boundary, not a complete removal of legacy code.

The old [split proposal](rpi_manual_control_split_boundary.md) describes an
intermediate serial-command-producing Pi design, not the current topology.

## Verification

Use fake GPIO/ADC, synthetic progressing panel/consumer state and bounded
clock advancement for policy tests. No hardware service is needed to test lamp
hysteresis, unreadable contacts, manual exception, stale status or menu direction.
Electrical settling, physical switch behavior and real E-stop/motor response
still require separate bench acceptance.

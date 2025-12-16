# RS485 gimbal bring-up checklist

Use this checklist when enabling the physical MKS SERVO42D/57D_RS485 gimbal on
the Jetson. The goal is to confirm that the serial bus is wired correctly, fill
out the `gimbal` block in `configs/dev.yaml`, and validate that commands and
encoder reads work via the CLI exerciser before the bridge enters the main loop.

## 1. Wire and power the RS485 bus
- Jetson UART: use the 3.3 V TTL UART (`/dev/ttyTHS0` by default) and route it
  through your TTL↔RS485 transceiver (A/B differential pairs) to the motor
  controllers. You do **not** need to enable `serial.rs485` mode in Python
  because the hardware transceiver handles the conversion.
- Motor power: ensure both axes are powered and share a common ground with the
  Jetson UART.
- Direction bits: in the motor menu, set pitch motor A "Dir" to CW and pitch
  motor B "Dir" to CCW so group commands spin them in opposite mechanical
  directions.

## 2. Fill in `configs/dev.yaml` → `gimbal`
Update the `gimbal` section so the bridge knows how to talk to your motors. Key
fields include:

- `serial_port` and `baudrate`: Jetson UART path (e.g., `/dev/ttyTHS0`) and
  RS485 baud (defaults to 38400).
- `yaw_addr`, `yaw_group_addr`: individual and optional group address for the
  yaw motor. Leave `yaw_group_addr` null if the yaw axis is solo.
- `pitch_group_addr`, `pitch_motor_a_addr`, `pitch_motor_b_addr`:
  - `pitch_group_addr` is the shared address used for group speed commands.
  - `pitch_motor_a_addr` / `pitch_motor_b_addr` stay unique for encoder reads
    and diagnostics.
- `pitch_encoder_authority`: choose `"a"` or `"b"` depending on which motor's
  encoder you trust when both are connected.
- `counts_per_rev`, `yaw_gear_ratio`, `pitch_gear_ratio`: map encoder counts to
  radians; set gear ratios to your drivetrain reduction.
- `yaw_accel_byte` / `pitch_accel_byte`: acceleration byte written into the F6
  command when commanding speeds.
- `yaw_rate_limit_rad_s` / `pitch_rate_limit_rad_s`: clamp the commanded rates
  before they are translated into motor speeds.
- `use_group_writes`: toggle to disable group addressing during bring-up if
  individual writes are more reliable on your wiring.
- `pitch_divergence_thresh_rad`: threshold for warning if the non-authoritative
  pitch encoder drifts from the primary reading.

## 3. Install Jetson extras
Install the Jetson extra dependencies so `pyserial` is available:

```bash
pip install -e .[jetson]
```

## 4. Validate the serial link with the CLI
Run the CLI exerciser **before** relying on the bridge. Use the same serial port
and addresses configured above.

- Command a temporary speed for the yaw motor:

```bash
python -m jetson.tools.test_mks_gimbal_serial speed --port /dev/ttyTHS0 --addr 1 --omega 0.5 --acc 10 --duration 2
```

- Read the encoder (counts and radians) from a motor:

```bash
python -m jetson.tools.test_mks_gimbal_serial read-enc --port /dev/ttyTHS0 --addr 1
```

- Issue an emergency stop if the motion is unexpected:

```bash
python -m jetson.tools.test_mks_gimbal_serial estop --port /dev/ttyTHS0 --addr 1
```

If these commands succeed without CRC or framing errors, the RS485 link is
healthy and ready for the gimbal bridge to consume `ControlCmd` messages.

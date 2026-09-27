"""F5 step 1 (read-only): yaw status and 0x31 counts. Sends no motion or enable."""

import json
import time

from common.gimbal.mks_servo42_rs485 import MksServo42Axis, RS485Bus

YAW = 1

with RS485Bus("/dev/ttyCH341USB0", baudrate=38400, timeout=0.12, max_retries=2) as bus:
    yaw = MksServo42Axis(bus, addr=YAW)
    for index in range(5):
        print(json.dumps({
            "sample": index,
            "monotonic_ns": time.monotonic_ns(),
            "status": yaw.status(),        # F1
            "counts": yaw.read_axis_counts(),  # 0x31, 48-bit signed
        }), flush=True)
        time.sleep(0.2)

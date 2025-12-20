import smbus
import time
import serial

# ===============================
# PCF8591 joystick ADC
# ===============================
bus = smbus.SMBus(1)
ADC_ADDR = 0x48

def read_adc(ch):
    ctrl = 0x40 | ch
    bus.write_byte(ADC_ADDR, ctrl)
    bus.read_byte(ADC_ADDR)
    return bus.read_byte(ADC_ADDR)

# ===============================
# RS485 serial
# ===============================
ser = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=38400,
    bytesize=8,
    parity="N",
    stopbits=1,
    timeout=0.05
)

def checksum(data):
    return sum(data) & 0xFF

# ===============================
# F6 — Speed Mode Command
# ===============================
def send_speed(addr, speed_rpm, direction, acc=5):
    """
    speed_rpm: 0 ~ 3000
    direction: 0 = CW, 1 = CCW
    acc: acceleration 0~255
    """
    # Speed encoding（高位在 byte4, 低位在 byte5 的低 4bits）
    speed = speed_rpm & 0x0FFF

    byte4 = ((direction & 1) << 7) | ((speed >> 8) & 0x0F)
    byte5 = speed & 0xFF

    packet = [
        0xFA,
        addr,
        0xF6,
        byte4,
        byte5,
        acc & 0xFF
    ]
    packet.append(checksum(packet))
    ser.write(bytes(packet))


# ===============================
# 搖桿 → RPM 轉換
# ===============================
def map_value_to_speed(value, deadzone=8, max_rpm=800):
    center = 128
    diff = value - center

    if abs(diff) < deadzone:
        return 0, 0  # dir, speed

    direction = 1 if diff > 0 else 0
    speed = int((abs(diff) / 128.0) * max_rpm)
    return direction, speed


# ===============================
# 主程式
# ===============================
def main():
    print("搖桿控制 RS485 三軸（X=addr1, Y=addr2+3）啟動...")

    while True:
        # 讀取搖桿
        joy_x = read_adc(0)
        joy_y = read_adc(1)

        # X 軸速度
        dir_x, spd_x = map_value_to_speed(joy_x)

        # Y 軸速度（兩顆同步）
        dir_y, spd_y = map_value_to_speed(joy_y)

        # ========== X 軸（addr=1）==========
        send_speed(1, spd_x, dir_x)

        # ========== Y 軸（兩顆，但方向相反）==========
        # 左馬達（addr=2）
        send_speed(2, spd_y, dir_y)

        # 右馬達（addr=3）→ 方向反轉
        send_speed(3, spd_y, 1 - dir_y)

        print(f"X: raw={joy_x}, rpm={spd_x}, dir={dir_x} | "
              f"Y: raw={joy_y}, rpm={spd_y}, dir(L/R)={dir_y}/{1 - dir_y}")

        time.sleep(0.05)


if __name__ == "__main__":
    main()

import RPi.GPIO as GPIO
import time
import sys

# ================= 腳位對應 =================
S  = 24   # Active-Low，press-only toggle（啟動/關閉）
S1 = 22   # Active-High，手動控制
S2 = 23   # Active-High，緊急停止（電平鎖定）

L1 = 17   # 緊急模式指示（低態繼電器）：LOW=緊急
L2 = 27   # 啟動模式繼電器：LOW=ACTIVE
J  = 26   # Jetson 控制：HIGH=啟動，LOW=停止
OUT25 = 25  # 手動輸出

# ================= 參數 =================
POLL_DT = 0.005
DEBOUNCE_S = 0.05

# ================= 狀態 =================
active = False
emergency = False
saved_active = False

prev_s_press = GPIO.HIGH
last_s_ts = 0.0

prev_in = {}

# ================= GPIO 初始化 =================
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(S,  GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(S1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(S2, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

GPIO.setup(L1, GPIO.OUT, initial=GPIO.HIGH)   # 正常
GPIO.setup(L2, GPIO.OUT, initial=GPIO.HIGH)   # 未啟動
GPIO.setup(J,  GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(OUT25, GPIO.OUT, initial=GPIO.HIGH)

INPUTS = [S, S1, S2]
prev_in = {p: GPIO.input(p) for p in INPUTS}

_last_out = {L1: GPIO.HIGH, L2: GPIO.HIGH, J: GPIO.LOW, OUT25: GPIO.HIGH}

def set_out(pin, level):
    if _last_out.get(pin) != level:
        GPIO.output(pin, level)
        _last_out[pin] = level
        print(f"[OUTPUT] GPIO{pin} set -> {level}")

def apply_normal_outputs():
    """非緊急模式下的輸出規則"""
    # L1：永遠不動（只給緊急模式用）
    set_out(L1, GPIO.HIGH)

    # ACTIVE 控制
    set_out(L2, GPIO.LOW if active else GPIO.HIGH)
    set_out(J,  GPIO.HIGH if active else GPIO.LOW)

    # OUT25
    if not active:
        set_out(OUT25, GPIO.HIGH)
    else:
        s1 = GPIO.input(S1)
        set_out(OUT25, GPIO.LOW if s1 == GPIO.HIGH else GPIO.HIGH)

def enter_emergency():
    global emergency, saved_active
    if emergency:
        return
    emergency = True
    saved_active = active
    print("[EMERGENCY] triggered")

    set_out(L1, GPIO.LOW)
    set_out(J, GPIO.LOW)
    set_out(OUT25, GPIO.HIGH)

def maintain_emergency():
    set_out(L1, GPIO.LOW)
    set_out(J, GPIO.LOW)
    set_out(OUT25, GPIO.HIGH)

def exit_emergency():
    global emergency, active
    emergency = False
    active = saved_active
    print("[EMERGENCY] released")
    apply_normal_outputs()

# ================= 主迴圈 =================
try:
    apply_normal_outputs()

    while True:
        s  = GPIO.input(S)
        s1 = GPIO.input(S1)
        s2 = GPIO.input(S2)

        for p, v in [(S, s), (S1, s1), (S2, s2)]:
            if v != prev_in[p]:
                print(f"[INPUT ] GPIO{p} changed -> {v}")
                prev_in[p] = v

        # 緊急模式（最高優先）
        if s2 == GPIO.HIGH:
            enter_emergency()
        else:
            if emergency:
                exit_emergency()

        if emergency:
            maintain_emergency()
            time.sleep(POLL_DT)
            continue

        # S：press-only toggle（非緊急）
        now = time.monotonic()
        if prev_s_press == GPIO.HIGH and s == GPIO.LOW:
            if now - last_s_ts >= DEBOUNCE_S:
                last_s_ts = now
                active = not active
                print(f"[LOGIC ] ACTIVE -> {active}")
                apply_normal_outputs()

        prev_s_press = s

        apply_normal_outputs()
        time.sleep(POLL_DT)

except KeyboardInterrupt:
    print("\nExit program")

finally:
    GPIO.output(L1, GPIO.HIGH)
    GPIO.output(L2, GPIO.HIGH)
    GPIO.output(J, GPIO.LOW)
    GPIO.output(OUT25, GPIO.HIGH)
    GPIO.cleanup()
    sys.exit(0)

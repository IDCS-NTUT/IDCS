"""Quick viewer for the placeholder simulation camera output."""

from __future__ import annotations

import cv2

from pc.sim_camera import SimCamera


def main() -> None:
    cam = SimCamera(width=640, height=480)
    while True:
        ok, frame = cam.next_frame()
        if not ok:
            break
        cv2.imshow("Sim preview", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

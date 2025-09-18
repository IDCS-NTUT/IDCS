# tools/gl_preview.py  (PC)
import cv2, numpy as np, time
from pc.sim_camera import SimCamera

cam = SimCamera(width=1280, height=720, renderer_name="gl")
for i in range(120):
    ok, frame = cam.next_frame()
    assert ok and frame is not None, "SimCamera returned no frame"
    cv2.imshow("GL preview", frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()

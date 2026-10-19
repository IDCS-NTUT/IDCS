Frames from the 2026-10-19 camera ROI bench run (run 03), decoded from the PC
recording of the annotated return stream (H.264, 20 Mbps). Overlay text sits
top-left and bottom-left only. Live seeds that were refused:
- camera_hand_thumb_1280x720.png: seed box x=426 y=281 w=176 h=145 (thumb-up hand inside)
- camera_hand_open_1280x720.png:  same box (open hand inside)
Offline native results on the Jetson (codex/camera-roi 86aa0cd):
- hand box 426,281,176,145 -> "seed lacks texture" (live: 0 and 2 seed features)
- desk equipment box 390,490,120,60 (thumb frame) -> "seed lacks texture"
- top-left box 20,0,120,200 (feature view origin 0,0) -> accepted ("operator seed", acquiring)
OpenCV goodFeaturesToTrack in the hand box: 64 (quality 0.01) / 47 (quality 0.05).

# IMX219 ISP override (not installed)

`camera_overrides.isp` is the camera module vendor's NVIDIA ISP override
(Char-lite calibration 1.9.0, April 2019: optical black, lens shading, AWB
grey line, colour matrix). It is kept for reference and **not installed**.

Evaluated on the Jetson (JetPack 6, 2026-09-29) by installing it as
`/var/nvidia/nvcam/settings/camera_overrides.isp` and restarting
`nvargus-daemon`:
- the daemon loads it but rejects ~25 attributes as invalid for this JetPack
  (companding, tone tables, exposure presets, sharpness/saturation/noise
  strengths, AWB night mode, ...);
- the output is unchanged: white-wall colour ratios under auto and
  fluorescent white balance and lens vignetting (edge/centre 0.79-0.80) match
  the stock `imx219.nito` tuning to two decimals.

The magenta cast comes from auto white balance choosing the wrong
illuminant; a fixed preset (`deepstream.argus_wb_mode`, fluorescent indoors,
daylight outdoors) corrects it. To install anyway:
`sudo install -m 664 camera_overrides.isp /var/nvidia/nvcam/settings/ &&
sudo systemctl restart nvargus-daemon`.

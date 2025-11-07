# DeepStream 6.3 prerequisites

This document captures the baseline required to bring the Jetson server onto the
DeepStream SDK while keeping x86_64 development hosts in sync. It focuses on the
6.3.0 release so we can migrate the inference pipeline without surprises from
mixed driver or library versions.

## Release alignment

| Component                | Version / Notes                              |
|--------------------------|----------------------------------------------|
| DeepStream SDK           | 6.3.0 (`deepstream-app --version-all`)       |
| JetPack / L4T (Jetson)   | JetPack 5.1.2 / L4T 35.4.1 (ships CUDA 11.4) |
| CUDA runtime             | 11.4 (from JetPack 5.1.2)                    |
| TensorRT                 | 8.5 (bundled with DeepStream 6.3)            |
| cuDNN                    | 8.6 (bundled with JetPack 5.1.2)             |

DeepStream 6.3 is the last release targeting JetPack 5.x. All Jetson devices on
JetPack 4.x must be upgraded before adopting the DeepStream pipeline.

## Supported hardware

DeepStream 6.3 runs on the following hardware in our fleet:

- **Jetson Xavier NX** – current production target. Requires JetPack 5.1.2 (or
  newer in the 5.x family) with the NVIDIA-provided DeepStream 6.3 package.
- **Jetson AGX Xavier / Jetson Orin** – optional expansion targets. JetPack
  5.1.2 images for these platforms ship the same CUDA/TensorRT stack so the
  DeepStream configuration can be reused.
- **x86_64 development host** – Ubuntu 20.04 with an NVIDIA RTX/Tesla GPU
  (Turing or newer). Install the 525+ production driver along with CUDA 11.8 and
  the DeepStream 6.3 x86_64 tarball if you need to prototype pipelines away from
  the Jetson hardware.

> The repo is private; download DeepStream artifacts directly from NVIDIA
> Developer using your corporate credentials rather than redistributing them.

## Installation (Jetson)

1. **Update JetPack** to 5.1.2 (L4T 35.4.1). The NVIDIA SDK Manager handles the
   OS image and base CUDA/cuDNN/TensorRT stack for Xavier NX, AGX Xavier, and
   Orin devices.
2. **Install DeepStream 6.3** using the `deepstream-6.3_6.3.0-1_arm64.deb`
   package from NVIDIA. Transfer it to the Jetson and run:
   ```bash
   sudo apt install ./deepstream-6.3_6.3.0-1_arm64.deb
   sudo /opt/nvidia/deepstream/deepstream/user_additional_install.sh
   ```
3. **Verify the install** by activating the Jetson virtual environment and
   running `deepstream-app --version-all`. The expected output is:
   ```
   deepstream-app version 6.3.0
   DeepStreamSDK 6.3.0
   CUDA Driver Version: 11.4
   CUDA Runtime Version: 11.4
   TensorRT Version: 8.5
   cuDNN Version: 8.6
   libNVWarp360 Version: 2.0.1d3
   ```

## Installation (x86_64 optional)

1. Install Ubuntu 20.04 LTS and the NVIDIA 525+ production driver.
2. Install CUDA 11.8 (the minimum required for DeepStream 6.3 on discrete GPUs).
3. Download the `deepstream_sdk_v6.3.0_x86_64.tbz2` package and extract it to
   `/opt/nvidia/deepstream/deepstream-6.3/`.
4. Run the `sudo ./install.sh` script from the extracted directory, then add the
   DeepStream GStreamer plugins to `GST_PLUGIN_PATH` as shown below.

## Python bindings (`pyds`)

The DeepStream Python bindings are distributed as version-specific wheels:

- Jetson (Python 3.8):
  ```bash
  pip install /opt/nvidia/deepstream/deepstream/lib/python/3.8/dist-packages/pyds-1.1.6-py3-none-linux_aarch64.whl
  ```
- x86_64 (Python 3.8/3.9):
  ```bash
  pip install /opt/nvidia/deepstream/deepstream/lib/python/3.8/dist-packages/pyds-1.1.6-py3-none-linux_x86_64.whl
  ```

After installation, export the following environment variables before running
the DeepStream-enabled Jetson server:

```bash
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:$LD_LIBRARY_PATH"
export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins"
export PYTHONPATH="/opt/nvidia/deepstream/deepstream/lib/python/3.8/dist-packages:$PYTHONPATH"
```

Add the exports to the Jetson service unit (or shell profile) so they are set
for every launch.

## Next steps

- Thread a `deepstream` section into `configs/dev.yaml` so runtime switches can
  discover the SDK path, inference configuration, and GPU ID.
- Build a minimal DeepStream Python sample under `jetson/` to confirm `pyds`
  imports and the `Gst.Pipeline` primitives before rewiring the production
  server.
- Update the operator runbook with any environment-specific notes (e.g., proxy
  settings for NVIDIA downloads) as they surface during migration.

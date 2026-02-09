#!/usr/bin/env bash
# libcamera-based streamer for Raspberry Pi OS (Bullseye/Bookworm)
# Uses libcamera-vid to capture and pipe into GStreamer for RTP streaming.

set -euo pipefail

JETSON_IP=${1:-192.168.0.5}
JETSON_PORT=${2:-5000}
WIDTH=${3:-1280}
HEIGHT=${4:-720}
FPS=${5:-30}
BITRATE_KBPS=${6:-4000}

echo "Streaming via libcamera to ${JETSON_IP}:${JETSON_PORT} at ${WIDTH}x${HEIGHT}@${FPS} (${BITRATE_KBPS} kbps)"

# libcamera-vid produces H.264 to stdout; pipe into gst-launch to RTP
libcamera-vid -t 0 --inline --width ${WIDTH} --height ${HEIGHT} --framerate ${FPS} -o - \
  | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=${JETSON_IP} port=${JETSON_PORT} sync=false async=false

#!/usr/bin/env bash
# Simple GStreamer RTP streamer for a CSI camera on a Raspberry Pi
# Streams H.264 RTP to a Jetson receiver (udpsink -> Jetson GRecv)

set -euo pipefail

JETSON_IP=${1:-192.168.0.5}
JETSON_PORT=${2:-5000}
WIDTH=${3:-1280}
HEIGHT=${4:-720}
FPS=${5:-30}
BITRATE_KBPS=${6:-4000}

echo "Streaming to ${JETSON_IP}:${JETSON_PORT} at ${WIDTH}x${HEIGHT}@${FPS} (${BITRATE_KBPS} kbps)"

# Try to use v4l2src (works for many Pi camera setups) with x264enc for H.264
gst-launch-1.0 -v \
  v4l2src do-timestamp=true ! video/x-raw,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1 ! \
  videoconvert ! x264enc bitrate=${BITRATE_KBPS} speed-preset=ultrafast tune=zerolatency byte-stream=true key-int-max=30 ! \
  h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=${JETSON_IP} port=${JETSON_PORT} sync=false async=false

# If your Pi provides a hardware encoder (omx or v4l2h264enc), replace x264enc with the appropriate encoder.

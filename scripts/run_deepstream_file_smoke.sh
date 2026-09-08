#!/usr/bin/env bash
# Isolated DeepStream nvinfer smoke test. It does not start IDCS control/ZMQ.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
input_file=${1:?"usage: $0 /absolute/path/to/h264-mp4 [--nvsort] [--paced] [--log-dir DIR]"}
shift
tracker_mode=
paced=false
log_dir=
while (($#)); do
  case "$1" in
    --nvsort)
      tracker_mode=--nvsort
      ;;
    --paced)
      paced=true
      ;;
    --log-dir)
      log_dir=${2:?"--log-dir requires a directory"}
      shift
      ;;
    *)
      echo "unsupported argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done
config_path="$repo_root/configs/deepstream/nvinfer_yolo26n_960.txt"

if [[ ! -f "$input_file" ]]; then
  echo "input file not found: $input_file" >&2
  exit 2
fi

tracker=()
if [[ "$tracker_mode" == "--nvsort" ]]; then
  ds_root=${DEEPSTREAM_ROOT:-/opt/nvidia/deepstream/deepstream}
  tracker=(
    ! nvtracker
    "ll-lib-file=$ds_root/lib/libnvds_nvmultiobjecttracker.so"
    "ll-config-file=$ds_root/samples/configs/deepstream-app/config_tracker_NvSORT.yml"
    tracker-width=640 tracker-height=384
  )
elif [[ -n "$tracker_mode" ]]; then
  echo "unsupported tracker mode: $tracker_mode (expected --nvsort)" >&2
  exit 2
fi

source_pacer=()
if [[ "$paced" == true ]]; then
  # The decoder's PTS values pace replay at the source's recorded 60 FPS.
  source_pacer=(identity sync=true !)
fi

if [[ -n "$log_dir" ]]; then
  mkdir -p "$log_dir"
  tegrastats --interval 1000 >"$log_dir/tegrastats.log" 2>&1 &
  tegrastats_pid=$!
  trap 'kill "$tegrastats_pid" 2>/dev/null || true' EXIT
fi

cd "$repo_root"
# Replay files may contain B-frames, so retain the decoded-picture buffer.
# Block replay queues instead of dropping compressed reference frames or hiding
# overload by silently discarding decoded frames. Live pipelines use a separate
# latest-only policy at safe frame boundaries.
gst-launch-1.0 -e \
  filesrc location="$input_file" ! qtdemux name=demux \
  demux.video_0 ! queue max-size-buffers=2 ! h264parse ! \
  nvv4l2decoder enable-max-performance=1 ! \
  "${source_pacer[@]}" \
  queue max-size-buffers=2 ! mux.sink_0 \
  nvstreammux name=mux batch-size=1 width=1280 height=720 live-source=false \
    batched-push-timeout=16666 ! \
  nvinfer config-file-path="$config_path" \
  "${tracker[@]}" \
  ! nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" ! \
  nvdsosd ! fpsdisplaysink text-overlay=false fps-update-interval=1000 \
    video-sink=fakesink sync=false

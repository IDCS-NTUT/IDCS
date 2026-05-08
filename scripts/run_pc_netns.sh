#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Run the PC streamer/UI inside a private Linux network namespace.

This wrapper moves one dedicated interface into the namespace and then runs
scripts/run_pc.sh there. The namespace only sees loopback plus that interface,
so IDCS UDP/ZMQ traffic cannot route through other host interfaces.

Required environment:
  IDCS_PC_IFACE=IFACE        dedicated interface to move into the namespace

Common optional environment:
  IDCS_PC_ADDR=ADDR/CIDR     address to assign inside the namespace
  IDCS_PC_GW=ADDR            default gateway inside the namespace
  IDCS_PC_NETNS=NAME         namespace name (default: idcs-pc)
  PYTHON=/path/to/python      Python executable for scripts/run_pc.sh
  IDCS_PC_RUN_USER=USER      user to run PC tools as (default: sudo caller)

Examples:
  sudo -E IDCS_PC_IFACE=enp3s0 IDCS_PC_ADDR=192.168.0.1/24 \
    bash scripts/run_pc_netns.sh configs/network.yaml configs/perception.yaml,configs/control.yaml,configs/system.yaml

  sudo -E IDCS_PC_IFACE=enp3s0 IDCS_PC_ADDR=192.168.0.1/24 IDCS_PC_GW=192.168.0.5 \
    bash scripts/run_pc_netns.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

if [[ "${EUID}" -ne 0 ]]; then
	echo "run_pc_netns.sh must run as root so it can create netns and move interfaces." >&2
	echo "Try: sudo -E IDCS_PC_IFACE=<iface> IDCS_PC_ADDR=192.168.0.1/24 bash scripts/run_pc_netns.sh" >&2
	exit 2
fi

IFACE=${IDCS_PC_IFACE:-}
if [[ -z "${IFACE}" ]]; then
	echo "IDCS_PC_IFACE is required; choose the dedicated Jetson/IDCS interface." >&2
	exit 2
fi

NETNS=${IDCS_PC_NETNS:-idcs-pc}
ADDR=${IDCS_PC_ADDR:-}
GATEWAY=${IDCS_PC_GW:-}
CONFIG_PATH=${1:-configs/network.yaml}
EXTRA_PATH=${2:-configs/perception.yaml,configs/control.yaml,configs/system.yaml}
RUN_USER=${IDCS_PC_RUN_USER:-${SUDO_USER:-}}

CREATED_NETNS=0
MOVED_IFACE=0

cleanup() {
	local status=$?
	set +e
	if [[ "${MOVED_IFACE}" -eq 1 ]]; then
		ip -n "${NETNS}" link set "${IFACE}" down 2>/dev/null || true
		ip -n "${NETNS}" link set "${IFACE}" netns 1 2>/dev/null || true
	fi
	if [[ "${CREATED_NETNS}" -eq 1 ]]; then
		ip netns delete "${NETNS}" 2>/dev/null || true
	fi
	exit "${status}"
}
trap cleanup EXIT INT TERM

if ! command -v ip >/dev/null 2>&1; then
	echo "The 'ip' command from iproute2 is required." >&2
	exit 2
fi

if ! ip netns list | awk '{print $1}' | grep -Fxq "${NETNS}"; then
	ip netns add "${NETNS}"
	CREATED_NETNS=1
fi

ip -n "${NETNS}" link set lo up

if ip link show dev "${IFACE}" >/dev/null 2>&1; then
	ip link set "${IFACE}" down
	ip link set "${IFACE}" netns "${NETNS}"
	MOVED_IFACE=1
elif ! ip -n "${NETNS}" link show dev "${IFACE}" >/dev/null 2>&1; then
	echo "Interface '${IFACE}' was not found in the host or '${NETNS}' namespace." >&2
	exit 2
fi

if [[ -n "${ADDR}" ]]; then
	ip -n "${NETNS}" addr flush dev "${IFACE}"
	ip -n "${NETNS}" addr add "${ADDR}" dev "${IFACE}"
fi

ip -n "${NETNS}" link set "${IFACE}" up

if [[ -n "${GATEWAY}" ]]; then
	ip -n "${NETNS}" route replace default via "${GATEWAY}" dev "${IFACE}"
fi

echo "[pc-netns] Namespace: ${NETNS}"
echo "[pc-netns] Interface: ${IFACE}"
ip -n "${NETNS}" -brief addr show dev "${IFACE}"
ip -n "${NETNS}" route show

ENV_ARGS=(
	PATH="${PATH:-}"
	PYTHON="${PYTHON:-python}"
	VIRTUAL_ENV="${VIRTUAL_ENV:-}"
	CONDA_PREFIX="${CONDA_PREFIX:-}"
	CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
	LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
	PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}"
	GI_TYPELIB_PATH="${GI_TYPELIB_PATH:-}"
	GST_PLUGIN_PATH="${GST_PLUGIN_PATH:-}"
	DISPLAY="${DISPLAY:-}"
	WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}"
	XAUTHORITY="${XAUTHORITY:-}"
	XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}"
)

CMD=(bash scripts/run_pc.sh "${CONFIG_PATH}" "${EXTRA_PATH}")
if [[ -n "${RUN_USER}" && "${RUN_USER}" != "root" ]]; then
	ip netns exec "${NETNS}" sudo -E -u "${RUN_USER}" env "${ENV_ARGS[@]}" "${CMD[@]}"
else
	ip netns exec "${NETNS}" env "${ENV_ARGS[@]}" "${CMD[@]}"
fi

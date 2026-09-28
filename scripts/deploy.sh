#!/usr/bin/env bash
# Deploy a tagged commit of main to the runtime checkouts and install units.
#
#   git tag deploy-N && scripts/deploy.sh deploy-N [pc,jetson,pi]
#
# For each host: fetch the tag from a git bundle into ~/Desktop/project/IDCS-runtime,
# check it out (refuses if the checkout has local changes), and install that
# host's unit files from deploy/systemd/<host>/ with a daemon-reload. Each fetch
# adds a full-history pack; git gc repacks once more than 8 accumulate. Nothing is
# restarted: restart the affected services yourself (docs/launch_procedure.md).
set -euo pipefail

TAG=${1:?usage: scripts/deploy.sh <tag> [pc,jetson,pi]}
HOSTS=${2:-pc,jetson,pi}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME=Desktop/project/IDCS-runtime
KEY=${IDCS_SSH_KEY:-$HOME/.ssh/id_ed25519_lan}
JETSON=${IDCS_JETSON:-idcs@192.168.0.5}
PI=${IDCS_PI:-idcs@192.168.0.3}

git -C "$REPO" rev-parse -q --verify "refs/tags/$TAG" >/dev/null || { echo "no tag $TAG in $REPO" >&2; exit 1; }
BUNDLE=$(mktemp --suffix=.bundle)
trap 'rm -f "$BUNDLE"' EXIT
git -C "$REPO" bundle create "$BUNDLE" main --tags 2>/dev/null

# Fetch and check out the tag in a runtime checkout; $1 = shell prefix to run it.
checkout='cd ~/'"$RUNTIME"' && [ -z "$(git status --porcelain)" ] || { echo "local changes in $(pwd), not deploying" >&2; exit 3; }
git fetch -q "$BUNDLE_PATH" "refs/tags/*:refs/tags/*" && git checkout -q "'"$TAG"'" &&
git -c gc.autoPackLimit=8 gc --auto --quiet && echo "$(hostname): $(git describe --tags)"'

if [[ ,$HOSTS, == *,pc,* ]]; then
  BUNDLE_PATH=$BUNDLE bash -c "$checkout"
  cp "$HOME/$RUNTIME"/deploy/systemd/pc/* "$HOME/.config/systemd/user/"
  systemctl --user daemon-reload
fi
if [[ ,$HOSTS, == *,jetson,* ]]; then
  scp -q -i "$KEY" "$BUNDLE" "$JETSON:/tmp/idcs-deploy.bundle"
  ssh -i "$KEY" "$JETSON" "BUNDLE_PATH=/tmp/idcs-deploy.bundle bash -c '$checkout' &&
    sudo cp ~/$RUNTIME/deploy/systemd/jetson/* /etc/systemd/system/ && sudo systemctl daemon-reload"
fi
if [[ ,$HOSTS, == *,pi,* ]]; then
  scp -q -i "$KEY" "$BUNDLE" "$PI:/tmp/idcs-deploy.bundle"
  ssh -i "$KEY" "$PI" "BUNDLE_PATH=/tmp/idcs-deploy.bundle bash -c '$checkout' &&
    cp ~/$RUNTIME/deploy/systemd/rpi/* ~/.config/systemd/user/ && systemctl --user daemon-reload"
fi
echo "deployed $TAG to $HOSTS; restart the affected services (docs/launch_procedure.md)"

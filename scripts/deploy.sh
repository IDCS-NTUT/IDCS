#!/usr/bin/env bash
# Explicit phases; no bulk runtime checkout or implicit motor activation.
# scripts/deploy.sh stage|validate|activate|verify|rollback|inventory|audit deploy-N [pc,jetson,pi]
set -euo pipefail
PHASE=${1:?phase required}
TAG=${2:?release-tag required}
HOSTS=${3:-pc,jetson,pi}
case "$PHASE" in stage|plan|validate|activate|verify|rollback|inventory|audit) ;; *) echo 'invalid phase' >&2; exit 2 ;; esac
[[ $TAG =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'invalid release id' >&2; exit 2; }
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=/home/idcs/Desktop/project
KEY=${IDCS_SSH_KEY:-$HOME/.ssh/id_ed25519_lan}
JETSON=${IDCS_JETSON:-idcs@192.168.0.5}
PI=${IDCS_PI:-idcs@192.168.0.3}
TASK_TMP=$(mktemp -d)
trap 'rm -f -- "$TASK_TMP/release.bundle" "$TASK_TMP/templates.tar"; rmdir -- "$TASK_TMP"' EXIT
if [[ $PHASE == stage ]]; then
  [[ -z $(git -C "$REPO" status --porcelain) ]] || { echo 'commit reviewed changes before staging' >&2; exit 3; }
  COMMIT=$(git -C "$REPO" rev-parse "refs/tags/$TAG^{commit}")
  [[ $COMMIT == $(git -C "$REPO" rev-parse HEAD) ]] || { echo 'stage from the tagged checkout so code and templates match' >&2; exit 3; }
  if [[ -n ${IDCS_BUNDLE_BASE:-} ]]; then
    git -C "$REPO" bundle create "$TASK_TMP/release.bundle" "$TAG" "^$IDCS_BUNDLE_BASE"
  else
    git -C "$REPO" bundle create "$TASK_TMP/release.bundle" "$TAG"
  fi
fi
tar -C "$REPO" -cf "$TASK_TMP/templates.tar" deploy/systemd
# IDCS_CONFIG_POLICY: reviewed config before/after hashes for this release, used by stage/plan.
POLICY=${IDCS_CONFIG_POLICY:-}
[[ -z $POLICY || -f $POLICY ]] || { echo "no config policy file $POLICY" >&2; exit 2; }
for HOST in ${HOSTS//,/ }; do
  case "$HOST" in pc) REMOTE=; PYTHON=$ROOT/venv/bin/python3 ;; jetson) REMOTE=$JETSON; PYTHON=$ROOT/bin/python3 ;; pi) REMOTE=$PI; PYTHON=$ROOT/bin/python3 ;; *) echo "unknown host: $HOST" >&2; exit 2 ;; esac
  DEST=$ROOT/deployments/managed/$TAG/installer
  POLICY_ARG=
  [[ -n $POLICY && ( $PHASE == stage || $PHASE == plan ) ]] && POLICY_ARG="--config-policy $DEST/config-policy.json"
  if [[ -z $REMOTE ]]; then
    mkdir -p "$DEST"
    cp "$REPO/tools/deployment.py" "$DEST/deployment.py"
    tar -xf "$TASK_TMP/templates.tar" -C "$DEST"
    [[ -n $POLICY_ARG ]] && cp "$POLICY" "$DEST/config-policy.json"
    if [[ $PHASE == stage ]]; then
      cp "$TASK_TMP/release.bundle" "$DEST/release.bundle"
      "$PYTHON" "$DEST/deployment.py" stage --role pc --release "$TAG" --templates "$DEST/deploy/systemd" --bundle "$DEST/release.bundle" --commit "$COMMIT" $POLICY_ARG
    else
      "$PYTHON" "$DEST/deployment.py" "$PHASE" --role pc --release "$TAG" --templates "$DEST/deploy/systemd" $POLICY_ARG
    fi
  else
    ssh -i "$KEY" "$REMOTE" "mkdir -p '$DEST'"
    scp -i "$KEY" "$REPO/tools/deployment.py" "$TASK_TMP/templates.tar" "$REMOTE:$DEST/"
    [[ -n $POLICY_ARG ]] && scp -i "$KEY" "$POLICY" "$REMOTE:$DEST/config-policy.json"
    ssh -i "$KEY" "$REMOTE" "tar -xf '$DEST/templates.tar' -C '$DEST'"
    if [[ $PHASE == stage ]]; then
      scp -i "$KEY" "$TASK_TMP/release.bundle" "$REMOTE:$DEST/release.bundle"
      ssh -i "$KEY" "$REMOTE" "'$PYTHON' '$DEST/deployment.py' stage --role '$HOST' --release '$TAG' --templates '$DEST/deploy/systemd' --bundle '$DEST/release.bundle' --commit '$COMMIT' $POLICY_ARG"
    else
      ssh -i "$KEY" "$REMOTE" "'$PYTHON' '$DEST/deployment.py' '$PHASE' --role '$HOST' --release '$TAG' --templates '$DEST/deploy/systemd' $POLICY_ARG"
    fi
  fi
done

"""Bind feature normalization to exact swarm model bytes, outside live YAML."""
import hashlib
import json
import math
from pathlib import Path

from common.config_constants import SWARM_EVAL_LEARNED_MODEL_NORMALIZATION


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_normalization(model_path):
    path = Path(model_path)
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("model_sha256") != sha256(path):
            raise ValueError(f"swarm model metadata digest mismatch: {path}")
        companions = metadata.get("companions", {})
    else:
        manifest = json.loads(path.with_name("normalization.json").read_text(encoding="utf-8"))
        if manifest.get("artifacts", {}).get(path.name) != sha256(path):
            raise ValueError(f"swarm model requires hash-bound normalization metadata: {path}")
        metadata = manifest
        companions = {name: checksum for name, checksum in manifest.get("artifacts", {}).items()
                      if name == path.name + ".data"}
    for name, checksum in companions.items():
        if Path(name).name != name or sha256(path.with_name(name)) != checksum:
            raise ValueError(f"swarm model companion digest mismatch: {path} / {name}")
    values = metadata["normalization"]
    expected = SWARM_EVAL_LEARNED_MODEL_NORMALIZATION
    if set(values) != set(expected) or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                                          for v in values.values()):
        raise ValueError(f"invalid swarm normalization metadata: {path}")
    return values


def bind_derived_model(model_path, source_path, *, normalization=None):
    """Call immediately after a reviewed export/build from this exact source."""
    model, source = Path(model_path), Path(source_path)
    if normalization is None:
        normalization = load_normalization(source)
    elif set(normalization) != set(SWARM_EVAL_LEARNED_MODEL_NORMALIZATION):
        raise ValueError("export source metadata must declare the complete normalization")
    metadata = dict(model_sha256=sha256(model), source_sha256=sha256(source),
                    source_name=source.name, normalization=normalization)
    if model.with_name(model.name + ".data").exists():
        companion = model.with_name(model.name + ".data")
        metadata["companions"] = {companion.name: sha256(companion)}
    from common.live_config import atomic_write
    atomic_write(model.with_suffix(model.suffix + ".metadata.json"),
                 json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata

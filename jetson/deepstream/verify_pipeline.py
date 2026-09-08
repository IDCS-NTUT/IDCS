"""Compatibility CLI for bounded DeepStream verification runs.

Pipeline construction and execution live in :mod:`jetson.deepstream.pipeline`
so the production runtime does not invoke a verifier module.
"""
from __future__ import annotations

from jetson.deepstream.pipeline import (
    StageClock,
    VerificationStats,
    _load_nvinfer_labels,
    _pipeline_description,
    _target_osd_suffix,
    run,
)

__all__ = ["run"]


if __name__ == "__main__":
    raise SystemExit(run())

"""Probe-owned policy swaps, independent of GStreamer and GPU bindings."""
from common.live_config import digest


def apply_pipeline_settings(stats, selector, values, *, shadow_factory, stitch_factory):
    shadow = values["deepstream"].get("shadow_tracks", {})
    stitching = values["deepstream"].get("id_stitching", {})
    policy = (shadow_factory(float(shadow["min_confidence"]), int(shadow["max_age_frames"]))
              if shadow.get("enabled") else None)
    gap = int(stitching.get("max_gap_frames", 45))
    if gap < 1:
        raise ValueError("deepstream.id_stitching.max_gap_frames must be positive")
    stitcher = stats.stitcher
    if stitching.get("enabled") and stitcher is None:
        stitcher = stitch_factory(max_gap_frames=gap)
    if selector is not None:
        selector.reconfigure(values, digest(values))
    stats.shadow_policy = policy
    if stitching.get("enabled"):
        stitcher._max_gap = gap
        stats.stitcher = stitcher
    else:
        stats.stitcher = None

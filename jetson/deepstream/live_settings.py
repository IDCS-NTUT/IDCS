"""Probe-owned policy swaps, independent of GStreamer and GPU bindings."""
from common.live_config import digest


def prepare_pipeline_settings(selector, values, *, shadow_factory, stitch_factory):
    shadow = values["deepstream"].get("shadow_tracks", {})
    stitching = values["deepstream"].get("id_stitching", {})
    policy = (shadow_factory(float(shadow["min_confidence"]), int(shadow["max_age_frames"]))
              if shadow.get("enabled") else None)
    gap = int(stitching.get("max_gap_frames", 45))
    if gap < 1:
        raise ValueError("deepstream.id_stitching.max_gap_frames must be positive")
    # Do not inspect/mutate probe-owned stitch history from this worker.
    stitcher = stitch_factory(max_gap_frames=gap) if stitching.get("enabled") else None
    selection = selector.prepare_settings(values, digest(values)) if selector is not None else None
    return policy, stitcher, gap, selection


def apply_pipeline_settings(stats, selector, prepared):
    policy, fresh_stitcher, gap, selection = prepared
    if selector is not None:
        selector.apply_settings(selection)
    stats.shadow_policy = policy
    if fresh_stitcher is not None:
        stitcher = stats.stitcher if stats.stitcher is not None else fresh_stitcher
        stitcher._max_gap = gap
        stats.stitcher = stitcher
    else:
        stats.stitcher = None

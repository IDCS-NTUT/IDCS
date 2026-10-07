"""Deterministic Cairo tile benchmark; no video, sockets, GPIO or motor owner."""
import argparse
import json
from pathlib import Path
import time

import cairo

from common.hud_telemetry import HudCache
from rpi.display.hud import PiHudRenderer
from tests.test_panel_hud import message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 30 <= args.frames <= 3000:
        parser.error('frames must be 30..3000')
    args.output.mkdir(parents=True, exist_ok=True)
    renderer = PiHudRenderer()
    cache = HudCache()
    values = message()
    elapsed = []
    images = []
    for i in range(args.frames):
        values['sequence'] = i + 1
        for name in ('pose', 'diagnostics', 'perception'):
            values[name]['sequence'] = i + 1
        values['pose']['data']['pan'] = .2 + i * .001
        values['perception']['data']['sequence'] = i + 1
        values['perception']['data']['tracks'][0]['box']['x'] = .55 + .025 * __import__('math').sin(i * .07)
        cache.accept(values, i / 30)
        began = time.perf_counter_ns()
        images = renderer.render(1280, 720, cache.view(i / 30 + .001, source_session='sim-A'))
        elapsed.append((time.perf_counter_ns() - began) / 1e6)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1280, 720)
    cr = cairo.Context(surface)
    cr.set_source_rgb(.16, .16, .16)
    cr.paint()
    for image in images:
        tile = cairo.ImageSurface.create_for_data(bytearray(image.data), cairo.FORMAT_ARGB32,
            image.width, image.height, image.stride)
        cr.set_source_surface(tile, image.x, image.y)
        cr.paint()
    surface.write_to_png(str(args.output / 'synthetic-hud.png'))
    timings = sorted(elapsed[10:])
    report = dict(frames=args.frames, width=1280, height=720, elements=renderer.elements,
        render_median_ms=timings[len(timings) // 2], render_p95_ms=timings[int(len(timings) * .95)],
        render_max_ms=max(timings), pixel_bytes_per_composition=sum(len(i.data) for i in images),
        qualification='Cairo build only; excludes Gst blending, real return-frame identity and network transit')
    (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

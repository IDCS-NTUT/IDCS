"""Make a YOLO ONNX see grey letterbox bars when nvinfer pads with black.

nvinfer letterboxes a non-square frame into the square network input with
black (0) bars and cannot use another colour. Ultralytics trains with grey
(114) bars; on the recorded fast-target clip black bars cost 7.5 points of
drone detection at threshold 0.30 (2026-10-04, migration journal). For a fixed
frame size the bars are always the same rows or columns, so this prepends

    images' = images * mask + (114/255) * (1 - mask)

to the graph, with mask = 1 inside the image area and 0 in the bars, then
writes a new ONNX to build the TensorRT engine from. Valid only for the frame
size it was built for, with nvinfer `maintain-aspect-ratio=1` and
`symmetric-padding=1`, and input already scaled to 0-1 (`net-scale-factor`
1/255).

    python tools/onnx_grey_letterbox.py model.onnx model_grey1280x720.onnx --frame 1280x720
"""
from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper

PAD = 114 / 255.0


def letterbox_rows_cols(frame_w: int, frame_h: int, net: int) -> tuple[int, int, int, int]:
    """(top, height, left, width) of the image inside the square input, as nvinfer places it."""
    scale = min(net / frame_w, net / frame_h)
    w, h = int(round(frame_w * scale)), int(round(frame_h * scale))
    return (net - h) // 2, h, (net - w) // 2, w


def add_grey_letterbox(model: onnx.ModelProto, frame_w: int, frame_h: int, pad: float = PAD) -> onnx.ModelProto:
    graph = model.graph
    inp = graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[2] != dims[3] or dims[2] <= 0:
        raise ValueError(f"expected a fixed square NCHW input, got {dims}")
    net = dims[2]
    top, h, left, w = letterbox_rows_cols(frame_w, frame_h, net)
    mask = np.zeros((1, 1, net, net), dtype=np.float32)
    mask[:, :, top:top + h, left:left + w] = 1.0
    bias = (pad * (1.0 - mask)).astype(np.float32)
    name = inp.name
    padded = f"{name}_grey_letterbox"
    for node in graph.node:
        node.input[:] = [padded if x == name else x for x in node.input]
    graph.initializer.extend([numpy_helper.from_array(mask, "grey_letterbox_mask"),
                              numpy_helper.from_array(bias, "grey_letterbox_bias")])
    nodes = [helper.make_node("Mul", [name, "grey_letterbox_mask"], [f"{padded}_masked"], name="grey_letterbox_mul"),
             helper.make_node("Add", [f"{padded}_masked", "grey_letterbox_bias"], [padded], name="grey_letterbox_add")]
    for node in reversed(nodes):
        graph.node.insert(0, node)
    model.metadata_props.append(onnx.StringStringEntryProto(
        key="idcs_grey_letterbox", value=f"frame={frame_w}x{frame_h} net={net} top={top} h={h} left={left} w={w} pad={pad:.6f}"))
    onnx.checker.check_model(model)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--frame", default="1280x720", help="camera/stream frame size WxH")
    args = ap.parse_args()
    fw, fh = (int(v) for v in args.frame.lower().split("x"))
    model = add_grey_letterbox(onnx.load(args.source), fw, fh)
    onnx.save(model, args.output)
    print(model.metadata_props[-1].value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

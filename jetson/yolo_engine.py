# TensorRT YOLO v8n engine loader (FP16 fixed 1x3x640x640)
import os, time, math
from typing import List, Tuple
import numpy as np
import cv2
from pycuda.compiler import SourceModule
import tensorrt as trt
import pycuda.autoinit        # noqa: F401
import pycuda.driver as cuda
from common.schemas import Box

# Nearest-neighbor
_CUDA_NEAREST = r"""
extern "C" __global__
void pack_letterbox_u8_bgr_to_nchw_f32(
    const unsigned char* __restrict__ src,  // H*W*3
    int src_w, int src_h, int src_stride,   // stride bytes = W*3
    float r, int left, int top,             // letterbox scale & offsets (dst space)
    float pad_r, float pad_g, float pad_b,  // pad color [0..1] RGB
    float* __restrict__ dst,                // NCHW float (1x3xSxS)
    int S                                   // dst side (e.g., 640)
){
    int x = blockIdx.x * blockDim.x + threadIdx.x; // dst x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // dst y
    if (x >= S || y >= S) return;

    // Map to source (nearest)
    float fx = (x - left) / r;
    float fy = (y - top ) / r;

    float R, G, B;
    if (fx < 0.f || fy < 0.f || fx > (float)(src_w - 1) || fy > (float)(src_h - 1)) {
        R = pad_r; G = pad_g; B = pad_b;
    } else {
        int sx = (int)(fx + 0.5f);
        int sy = (int)(fy + 0.5f);
        const unsigned char* p = src + sy * src_stride + sx * 3;
        unsigned char b = p[0], g = p[1], r8 = p[2];
        R = (float)r8 * (1.0f/255.0f);
        G = (float)g  * (1.0f/255.0f);
        B = (float)b  * (1.0f/255.0f);
    }
    int idx = y * S + x;
    dst[0*S*S + idx] = R;
    dst[1*S*S + idx] = G;
    dst[2*S*S + idx] = B;
}
""";

# Bilinear
_CUDA_BILINEAR = r"""
extern "C" __global__
void pack_letterbox_u8_bgr_to_nchw_f32_bilinear(
    const unsigned char* __restrict__ src,  // H*W*3 BGR
    int src_w, int src_h, int src_stride,
    float r, int left, int top,
    int roi_w, int roi_h,
    float pad_r, float pad_g, float pad_b,  // [0..1] RGB
    float* __restrict__ dst,
    int S)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // dst x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // dst y
    if (x >= S || y >= S) return;

    // Outside ROI => padding
    if (x < left || x >= left + roi_w || y < top || y >= top + roi_h) {
        int idx = y * S + x;
        dst[0*S*S + idx] = pad_r;
        dst[1*S*S + idx] = pad_g;
        dst[2*S*S + idx] = pad_b;
        return;
    }

    // Center-aligned bilinear
    float fx = ((x - left) + 0.5f) / r - 0.5f;
    float fy = ((y - top ) + 0.5f) / r - 0.5f;

    int sx = (int)floorf(fx);
    int sy = (int)floorf(fy);
    float tx = fx - (float)sx;
    float ty = fy - (float)sy;

    int sx0 = max(0, min(src_w - 1, sx));
    int sy0 = max(0, min(src_h - 1, sy));
    int sx1 = max(0, min(src_w - 1, sx + 1));
    int sy1 = max(0, min(src_h - 1, sy + 1));

    const unsigned char* p00 = src + sy0 * src_stride + sx0 * 3;
    const unsigned char* p10 = src + sy0 * src_stride + sx1 * 3;
    const unsigned char* p01 = src + sy1 * src_stride + sx0 * 3;
    const unsigned char* p11 = src + sy1 * src_stride + sx1 * 3;

    float R00 = (float)p00[2] * (1.0f/255.0f);
    float G00 = (float)p00[1] * (1.0f/255.0f);
    float B00 = (float)p00[0] * (1.0f/255.0f);
    float R10 = (float)p10[2] * (1.0f/255.0f);
    float G10 = (float)p10[1] * (1.0f/255.0f);
    float B10 = (float)p10[0] * (1.0f/255.0f);
    float R01 = (float)p01[2] * (1.0f/255.0f);
    float G01 = (float)p01[1] * (1.0f/255.0f);
    float B01 = (float)p01[0] * (1.0f/255.0f);
    float R11 = (float)p11[2] * (1.0f/255.0f);
    float G11 = (float)p11[1] * (1.0f/255.0f);
    float B11 = (float)p11[0] * (1.0f/255.0f);

    float w00 = (1.0f - tx) * (1.0f - ty);
    float w10 = tx * (1.0f - ty);
    float w01 = (1.0f - tx) * ty;
    float w11 = tx * ty;

    float R = R00*w00 + R10*w10 + R01*w01 + R11*w11;
    float G = G00*w00 + G10*w10 + G01*w01 + G11*w11;
    float B = B00*w00 + B10*w10 + B01*w01 + B11*w11;

    int idx = y * S + x;
    dst[0*S*S + idx] = R;
    dst[1*S*S + idx] = G;
    dst[2*S*S + idx] = B;
}
""";

# Lazy compile on first use
_mod_nearest = SourceModule(_CUDA_NEAREST, options=['-use_fast_math'])
_pack_nearest = _mod_nearest.get_function("pack_letterbox_u8_bgr_to_nchw_f32")

_mod_bilinear = SourceModule(_CUDA_BILINEAR, options=['-use_fast_math'])
_pack_bilinear = _mod_bilinear.get_function("pack_letterbox_u8_bgr_to_nchw_f32_bilinear")       
'''
def letterbox(img, new_shape=640, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, left, top

def nms(boxes, scores, classes, iou_thres=0.45, conf_thres=0.25):
    idx = scores >= conf_thres
    boxes, scores, classes = boxes[idx], scores[idx], classes[idx]
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return [(boxes[i], scores[i], int(classes[i])) for i in keep]
'''
class YoloEngine:
    def __init__(self, engine_path, conf_thres=0.25, iou_thres=0.45,
                 input_size=640, preprocess_mode="bilinear"):
        self.conf = conf_thres
        self.iou  = iou_thres
        self.sz   = int(input_size)
        self.preprocess_mode = preprocess_mode
        self._d_src = None
        self._d_src_size = 0
        # --- TensorRT load (outline) ---

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Bindings (names you just dumped)
        self.input_name  = "images"
        self.output_name = "output0"
        # device buffers

        in_shape  = (1, 3, self.sz, self.sz)
        out_shape = (1, 300, 6)
        # bytes = (#elements) * sizeof(float32)
        self.in_bytes  = int(np.prod(in_shape,  dtype=np.int64) * np.dtype(np.float32).itemsize)
        self.out_bytes = int(np.prod(out_shape, dtype=np.int64) * np.dtype(np.float32).itemsize)

        self.d_input  = cuda.mem_alloc(self.in_bytes)
        self.d_output = cuda.mem_alloc(self.out_bytes)

        # host buffer for output
        self.h_output = np.empty(out_shape, dtype=np.float32)

        # bindings (keep as ints)
        if hasattr(self.engine, "num_bindings"):
            self._use_tensor_api = False
            self.bindings = [None] * self.engine.num_bindings
            self.bindings[self.engine.get_binding_index(self.input_name)] = int(self.d_input)
            self.bindings[self.engine.get_binding_index(self.output_name)] = int(self.d_output)
        else:
            self._use_tensor_api = True
            self.bindings = [None] * self.engine.num_io_tensors
            self._tensor_indices = {
                self.engine.get_tensor_name(i): i
                for i in range(self.engine.num_io_tensors)
            }
            self.bindings[self._tensor_indices[self.input_name]] = int(self.d_input)
            self.bindings[self._tensor_indices[self.output_name]] = int(self.d_output)


    
    def infer(self, bgr_frame):
            if not isinstance(bgr_frame, np.ndarray):
                raise TypeError("bgr_frame must be a numpy.ndarray")
            if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
                raise ValueError("bgr_frame must have shape (H, W, 3)")
            if bgr_frame.dtype != np.uint8:
                bgr_frame = bgr_frame.astype(np.uint8, copy=False)
            if not bgr_frame.flags.c_contiguous:
                bgr_frame = np.ascontiguousarray(bgr_frame)

            H, W = bgr_frame.shape[:2]
            S = self.sz

            # --- letterbox math used by the CUDA preprocess kernels ---
            r = min(float(S) / H, float(S) / W)
            nh = int(round(H * r))
            nw = int(round(W * r))
            top  = (S - nh) // 2
            left = (S - nw) // 2

            # --- upload raw BGR and launch chosen kernel to write NCHW float into self.d_input ---
            src_bytes = H * W * 3
            if self._d_src is None or self._d_src_size < src_bytes:
                if self._d_src is not None:
                    self._d_src.free()
                self._d_src = cuda.mem_alloc(src_bytes)
                self._d_src_size = src_bytes
            cuda.memcpy_htod_async(self._d_src, bgr_frame, self.stream)

            block = (32, 16, 1)
            grid  = ((S + block[0] - 1) // block[0], (S + block[1] - 1) // block[1], 1)
            pad = np.float32(114.0/255.0)
            if self.preprocess_mode.lower() == "bilinear":
                _pack_bilinear(
                    self._d_src, np.int32(W), np.int32(H), np.int32(W*3),
                    np.float32(r), np.int32(left), np.int32(top),
                    np.int32(nw), np.int32(nh),
                    pad, pad, pad,
                    self.d_input, np.int32(S),
                    block=block, grid=grid, stream=self.stream
                )
            else:
                _pack_nearest(
                    self._d_src, np.int32(W), np.int32(H), np.int32(W*3),
                    np.float32(r), np.int32(left), np.int32(top),
                    pad, pad, pad,
                    self.d_input, np.int32(S),
                    block=block, grid=grid, stream=self.stream
                )

            # --- TensorRT inference ---
            if self._use_tensor_api and hasattr(self.context, "execute_async_v3"):
                self.context.set_tensor_address(self.input_name, int(self.d_input))
                self.context.set_tensor_address(self.output_name, int(self.d_output))
                self.context.execute_async_v3(self.stream.handle)
            else:
                self.context.execute_async_v2(self.bindings, self.stream.handle, None)

            # --- DtoH of output0 (1,300,6) ---
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            det = self.h_output[0]  # (300, 6)

            # Each row: [x1, y1, x2, y2, score, class] in 640x640 (letterboxed) space
            # Filter by score
            conf_mask = det[:, 4] >= self.conf
            det = det[conf_mask]
            if det.size == 0:
                return []

            # Map xyxy from letterboxed SxS back to original WxH, then normalize
            out = []
            for x1, y1, x2, y2, score, cls in det:
                # subtract padding
                x1 -= left; y1 -= top
                x2 -= left; y2 -= top
                # scale back from r
                x1 = x1 / r; x2 = x2 / r
                y1 = y1 / r; y2 = y2 / r
                # clip to image
                x1 = float(np.clip(x1, 0, W)); x2 = float(np.clip(x2, 0, W))
                y1 = float(np.clip(y1, 0, H)); y2 = float(np.clip(y2, 0, H))
                # convert to normalized xywh
                w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                out.append(
                    Box(
                        x = x1 / W,
                        y = y1 / H,
                        w = w  / W,
                        h = h  / H,
                        cls = str(int(cls)),
                        conf = float(score),
                    )
                )
            return out    

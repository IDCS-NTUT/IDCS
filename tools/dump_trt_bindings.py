from typing import List
import time
# tools/dump_trt_bindings.py
import tensorrt as trt, sys, json
assert len(sys.argv) == 2, "Usage: python dump_trt_bindings.py <engine.plan>"
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(sys.argv[1], "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine  = runtime.deserialize_cuda_engine(f.read())

info = []
# Old-style binding API (works on all TRT 7/8):
for i in range(engine.num_bindings):
    name  = engine.get_binding_name(i)
    shape = tuple(engine.get_binding_shape(i))
    dtype = str(engine.get_binding_dtype(i))
    io    = "input" if engine.binding_is_input(i) else "output"
    info.append({"index": i, "name": name, "io": io, "dtype": dtype, "shape": shape})

# Newer tensor API (TRT 8.5+), if available:
if hasattr(engine, "num_io_tensors"):
    info2 = []
    for i in range(engine.num_io_tensors):
        name  = engine.get_tensor_name(i)
        io    = "input" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output"
        dtype = str(engine.get_tensor_dtype(name))
        shape = tuple(engine.get_tensor_shape(name))
        info2.append({"index": i, "name": name, "io": io, "dtype": dtype, "shape": shape})
    print("\n# Tensor API:")
    print(json.dumps(info2, indent=2))

print("\n# Binding API:")
print(json.dumps(info, indent=2))


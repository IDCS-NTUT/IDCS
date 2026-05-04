"""TensorRT runtime wrapper for exported swarm policy engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional deployment dependency
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except ModuleNotFoundError:  # pragma: no cover - optional deployment dependency
    trt = None  # type: ignore[assignment]
    cuda = None  # type: ignore[assignment]

__all__ = ["SwarmPolicyTensorRTEngine", "TensorRTRuntimeUnavailableError"]


class TensorRTRuntimeUnavailableError(RuntimeError):
    """Raised when TensorRT runtime dependencies are unavailable."""


@dataclass(frozen=True)
class _TensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype


class SwarmPolicyTensorRTEngine:
    """Small TensorRT inference wrapper for swarm policy selection.

    The exported engine may be either:
    - `policy_only`
    - `policy_value`
    - `policy_value_class`
    """

    def __init__(self, engine_path: str | Path) -> None:
        if trt is None or cuda is None:
            raise TensorRTRuntimeUnavailableError(
                "TensorRT and PyCUDA are required for swarm TensorRT inference"
            )

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Swarm TensorRT engine not found: {self.engine_path}")

        self._logger = trt.Logger(trt.Logger.WARNING)
        with self.engine_path.open("rb") as handle:
            runtime = trt.Runtime(self._logger)
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context for {self.engine_path}")
        self.stream = cuda.Stream()

        self._tensor_api = hasattr(self.engine, "num_io_tensors")
        self._target_features_name: Optional[str] = None
        self._global_features_name: Optional[str] = None
        self._target_mask_name: Optional[str] = None
        self._policy_output_name: Optional[str] = None
        self._value_output_name: Optional[str] = None
        self._threat_class_output_name: Optional[str] = None
        self._collect_tensor_names()

        if self._target_features_name is None or self._global_features_name is None:
            raise RuntimeError(
                f"Could not identify swarm TensorRT inputs in {self.engine_path.name}"
            )
        if self._policy_output_name is None:
            raise RuntimeError(
                f"Could not identify policy logits output in {self.engine_path.name}"
            )

        input_shape = self._get_declared_shape(self._target_features_name)
        if len(input_shape) != 3:
            raise RuntimeError(
                f"Expected target_features input rank 3, got {input_shape} in {self.engine_path.name}"
            )
        global_shape = self._get_declared_shape(self._global_features_name)
        if len(global_shape) != 2:
            raise RuntimeError(
                f"Expected global_features input rank 2, got {global_shape} in {self.engine_path.name}"
            )

        target_profile_shape = self._get_profile_max_shape(self._target_features_name)
        global_profile_shape = self._get_profile_max_shape(self._global_features_name)

        self.target_feature_size = int(
            max(
                int(input_shape[2]) if input_shape[2] > 0 else 0,
                int(target_profile_shape[2]) if target_profile_shape[2] > 0 else 0,
            )
        )
        self.global_feature_size = int(
            max(
                int(global_shape[1]) if global_shape[1] > 0 else 0,
                int(global_profile_shape[1]) if global_profile_shape[1] > 0 else 0,
            )
        )
        self.max_targets = int(
            max(
                int(input_shape[1]) if input_shape[1] > 0 else 0,
                int(target_profile_shape[1]) if target_profile_shape[1] > 0 else 0,
            )
        )
        self.requires_target_mask = self._target_mask_name is not None

    @property
    def supports_value_head(self) -> bool:
        return self._value_output_name is not None

    @property
    def policy_output_name(self) -> str:
        if self._policy_output_name is None:
            raise RuntimeError("Policy output is not available")
        return self._policy_output_name

    @property
    def threat_class_output_name(self) -> Optional[str]:
        return self._threat_class_output_name

    def predict(
        self,
        target_features: np.ndarray,
        global_features: np.ndarray,
        target_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Run one forward pass and return named outputs."""
        if target_features.ndim != 3:
            raise ValueError(
                f"target_features must have shape [batch, num_targets, num_features], got {target_features.shape}"
            )
        if global_features.ndim != 2:
            raise ValueError(
                f"global_features must have shape [batch, num_global_features], got {global_features.shape}"
            )
        if target_features.shape[0] != global_features.shape[0]:
            raise ValueError("target_features and global_features batch size must match")
        if target_features.shape[2] != self.target_feature_size:
            raise ValueError(
                f"Expected {self.target_feature_size} target features, got {target_features.shape[2]}"
            )
        if global_features.shape[1] != self.global_feature_size:
            raise ValueError(
                f"Expected {self.global_feature_size} global features, got {global_features.shape[1]}"
            )
        if self.max_targets and target_features.shape[1] > self.max_targets:
            raise ValueError(
                f"Engine supports at most {self.max_targets} target slots, got {target_features.shape[1]}"
            )

        input_arrays: Dict[str, np.ndarray] = {
            self._target_features_name: self._coerce_input(
                self._target_features_name, target_features
            ),
            self._global_features_name: self._coerce_input(
                self._global_features_name, global_features
            ),
        }
        if self._target_mask_name is not None:
            if target_mask is None:
                raise ValueError("target_mask is required by this TensorRT engine")
            input_arrays[self._target_mask_name] = self._coerce_input(
                self._target_mask_name, target_mask
            )

        if self._tensor_api:
            return self._predict_tensor_api(input_arrays)
        return self._predict_binding_api(input_arrays)

    def _collect_tensor_names(self) -> None:
        if self._tensor_api:
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self._register_input_name(name)
                else:
                    self._register_output_name(name)
            return

        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            if self.engine.binding_is_input(index):
                self._register_input_name(name)
            else:
                self._register_output_name(name)

    def _register_input_name(self, name: str) -> None:
        lowered = name.lower()
        if "target_features" in lowered:
            self._target_features_name = name
        elif "global_features" in lowered:
            self._global_features_name = name
        elif "target_mask" in lowered:
            self._target_mask_name = name

    def _register_output_name(self, name: str) -> None:
        lowered = name.lower()
        if "threat_class" in lowered or "class_logits" in lowered:
            self._threat_class_output_name = name
        elif "policy" in lowered or "logit" in lowered:
            self._policy_output_name = name
        elif "value" in lowered:
            self._value_output_name = name

    def _get_declared_shape(self, name: str) -> Tuple[int, ...]:
        if self._tensor_api:
            return tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
        return tuple(int(dim) for dim in self.engine.get_binding_shape(self.engine.get_binding_index(name)))

    def _get_profile_max_shape(self, name: str) -> Tuple[int, ...]:
        if self._tensor_api:
            min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(name, 0)
            return tuple(int(dim) for dim in max_shape)
        binding_index = self.engine.get_binding_index(name)
        max_shape = self.engine.get_profile_shape(0, binding_index)[2]
        return tuple(int(dim) for dim in max_shape)

    def _trt_dtype_for_name(self, name: str):
        if self._tensor_api:
            return self.engine.get_tensor_dtype(name)
        return self.engine.get_binding_dtype(self.engine.get_binding_index(name))

    def _numpy_dtype_for_name(self, name: str) -> np.dtype:
        dtype = self._trt_dtype_for_name(name)
        return np.dtype(trt.nptype(dtype))

    def _coerce_input(self, name: str, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=self._numpy_dtype_for_name(name))
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return array

    def _set_input_shape(self, name: str, shape: Tuple[int, ...]) -> None:
        if self._tensor_api:
            self.context.set_input_shape(name, shape)
            return
        binding_index = self.engine.get_binding_index(name)
        self.context.set_binding_shape(binding_index, shape)

    def _current_output_shape(self, name: str) -> Tuple[int, ...]:
        if self._tensor_api:
            return tuple(int(dim) for dim in self.context.get_tensor_shape(name))
        binding_index = self.engine.get_binding_index(name)
        return tuple(int(dim) for dim in self.context.get_binding_shape(binding_index))

    def _predict_tensor_api(self, input_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for name, array in input_arrays.items():
            self._set_input_shape(name, tuple(int(dim) for dim in array.shape))

        buffers: Dict[str, Tuple[np.ndarray, cuda.DeviceAllocation]] = {}
        all_names = list(input_arrays.keys()) + [self._policy_output_name]
        if self._value_output_name is not None:
            all_names.append(self._value_output_name)
        if self._threat_class_output_name is not None:
            all_names.append(self._threat_class_output_name)

        for name in all_names:
            if name is None:
                continue
            if name in input_arrays:
                host_array = input_arrays[name]
            else:
                shape = self._current_output_shape(name)
                host_array = np.empty(shape, dtype=self._numpy_dtype_for_name(name))
            device_buffer = cuda.mem_alloc(host_array.nbytes)
            buffers[name] = (host_array, device_buffer)
            self.context.set_tensor_address(name, int(device_buffer))

        for name, (host_array, device_buffer) in buffers.items():
            if name in input_arrays:
                cuda.memcpy_htod_async(device_buffer, host_array, self.stream)

        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path.name}")

        outputs: Dict[str, np.ndarray] = {}
        for name, (host_array, device_buffer) in buffers.items():
            if name in input_arrays:
                continue
            cuda.memcpy_dtoh_async(host_array, device_buffer, self.stream)
            outputs[name] = host_array
        self.stream.synchronize()
        return outputs

    def _predict_binding_api(self, input_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        bindings = [0] * self.engine.num_bindings
        host_outputs: Dict[str, np.ndarray] = {}
        output_device_buffers: Dict[str, cuda.DeviceAllocation] = {}
        device_buffers = []

        for name, array in input_arrays.items():
            binding_index = self.engine.get_binding_index(name)
            self._set_input_shape(name, tuple(int(dim) for dim in array.shape))
            device_buffer = cuda.mem_alloc(array.nbytes)
            device_buffers.append(device_buffer)
            bindings[binding_index] = int(device_buffer)
            cuda.memcpy_htod_async(device_buffer, array, self.stream)

        output_names = [self._policy_output_name]
        if self._value_output_name is not None:
            output_names.append(self._value_output_name)
        if self._threat_class_output_name is not None:
            output_names.append(self._threat_class_output_name)
        for name in output_names:
            if name is None:
                continue
            binding_index = self.engine.get_binding_index(name)
            shape = self._current_output_shape(name)
            host_array = np.empty(shape, dtype=self._numpy_dtype_for_name(name))
            device_buffer = cuda.mem_alloc(host_array.nbytes)
            device_buffers.append(device_buffer)
            bindings[binding_index] = int(device_buffer)
            host_outputs[name] = host_array
            output_device_buffers[name] = device_buffer

        if not self.context.execute_async_v2(bindings, self.stream.handle, None):
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path.name}")

        for name, host_array in host_outputs.items():
            cuda.memcpy_dtoh_async(host_array, output_device_buffers[name], self.stream)
        self.stream.synchronize()
        return host_outputs

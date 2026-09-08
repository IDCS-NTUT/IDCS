// DeepStream parser for Ultralytics YOLO26 TensorRT exports with NMS included.
//
// Expected output tensor: float32 [1, 300, 6], each row [x1, y1, x2, y2,
// confidence, class_id] in the network-input coordinate system. DeepStream
// transforms these parser coordinates back to the source frame when aspect
// ratio preservation is enabled in the nvinfer configuration.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

constexpr int kRecordWidth = 6;
std::atomic<bool> first_call{true};

bool finite(float value) {
    return std::isfinite(value);
}

}  // namespace

extern "C" bool NvDsInferParseCustomYolo26(
    std::vector<NvDsInferLayerInfo> const& output_layers,
    NvDsInferNetworkInfo const& network_info,
    NvDsInferParseDetectionParams const& detection_params,
    std::vector<NvDsInferObjectDetectionInfo>& objects) {
    if (output_layers.size() != 1 || output_layers[0].buffer == nullptr ||
        network_info.width == 0 || network_info.height == 0) {
        return false;
    }

    const NvDsInferLayerInfo& output = output_layers[0];
    const auto* values = static_cast<const float*>(output.buffer);
    const std::size_t element_count = output.inferDims.numElements;
    const bool log_first_call = first_call.exchange(false);
    if (log_first_call) {
        std::fprintf(
            stderr,
            "[NvDsInferParseCustomYolo26] layers=%zu output=%s elements=%zu dims=%u network=%ux%u classes=%zu\n",
            output_layers.size(),
            output.layerName == nullptr ? "<unnamed>" : output.layerName,
            element_count,
            output.inferDims.numDims,
            network_info.width,
            network_info.height,
            detection_params.perClassPreclusterThreshold.size());
        std::fflush(stderr);
    }
    if (element_count == 0 || element_count % kRecordWidth != 0) {
        return false;
    }

    const std::size_t record_count = element_count / kRecordWidth;
    for (std::size_t index = 0; index < record_count; ++index) {
        const float* row = values + index * kRecordWidth;
        const float x1 = row[0];
        const float y1 = row[1];
        const float x2 = row[2];
        const float y2 = row[3];
        const float confidence = row[4];
        const float class_value = row[5];

        if (!finite(x1) || !finite(y1) || !finite(x2) || !finite(y2) ||
            !finite(confidence) || !finite(class_value)) {
            continue;
        }

        const int class_id = static_cast<int>(std::lround(class_value));
        if (class_id < 0 ||
            static_cast<std::size_t>(class_id) >= detection_params.perClassPreclusterThreshold.size() ||
            confidence < detection_params.perClassPreclusterThreshold[class_id]) {
            continue;
        }

        const float left = std::clamp(x1, 0.0F, static_cast<float>(network_info.width));
        const float top = std::clamp(y1, 0.0F, static_cast<float>(network_info.height));
        const float right = std::clamp(x2, 0.0F, static_cast<float>(network_info.width));
        const float bottom = std::clamp(y2, 0.0F, static_cast<float>(network_info.height));
        if (right <= left || bottom <= top) {
            continue;
        }

        NvDsInferObjectDetectionInfo detection{};
        detection.classId = static_cast<unsigned int>(class_id);
        detection.left = left;
        detection.top = top;
        detection.width = right - left;
        detection.height = bottom - top;
        detection.detectionConfidence = confidence;
        detection.rotation_angle = 0.0F;
        objects.push_back(detection);
    }
    if (log_first_call) {
        std::fprintf(
            stderr,
            "[NvDsInferParseCustomYolo26] parsed_objects=%zu\n",
            objects.size());
        std::fflush(stderr);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo26);

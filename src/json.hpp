#pragma once

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#include "mlx/ir.hpp"

namespace mlx::onnx {

using OrderedJson = nlohmann::ordered_json;

MLX_API OrderedJson ir_to_onnx_json_payload(
    const OrderedJson& payload,
    int64_t opset,
    const std::string& model_name);

MLX_API OrderedJson ir_compatibility_report_payload(const OrderedJson& payload);

MLX_API OnnxBinaryArtifact build_onnx_binary_artifact_from_stub(
    const OrderedJson& onnx_stub,
    const OnnxBinaryWriteOptions& options);

} // namespace mlx::onnx

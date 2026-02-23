#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/export.h"

namespace mlx::onnx {

constexpr int64_t kGraphIrVersion = 1;

using IrCaptureFunction = std::function<std::vector<mlx::core::array>(
    const mlx::core::Args&,
    const mlx::core::Kwargs&)>;

struct OnnxBinaryWriteOptions {
  bool external_data = false;
  std::string external_data_file = "weights.bin";
  int64_t external_data_size_threshold = 1024;
};

struct OnnxBinaryArtifact {
  std::string model_bytes;
  std::string external_data_bytes;
  bool has_external_data = false;
};

MLX_API std::string ir_to_onnx_json(
    const std::string& ir_json,
    int64_t opset,
    const std::string& model_name);

MLX_API std::string ir_to_onnx(
    const std::string& target_path,
    const std::string& ir_json,
    int64_t opset,
    const std::string& model_name,
    const OnnxBinaryWriteOptions& options);

MLX_API std::string ir_compatibility_report_json(
    const std::string& ir_json);

MLX_API std::string export_ir_json(
    const IrCaptureFunction& fun,
    const mlx::core::Args& args,
    const mlx::core::Kwargs& kwargs,
    bool shapeless);

MLX_API std::string export_onnx_compatibility_report_json(
    const IrCaptureFunction& fun,
    const mlx::core::Args& args,
    const mlx::core::Kwargs& kwargs,
    bool shapeless);

MLX_API std::string export_onnx_json(
    const IrCaptureFunction& fun,
    const mlx::core::Args& args,
    const mlx::core::Kwargs& kwargs,
    bool shapeless,
    int64_t opset,
    const std::string& model_name);

MLX_API std::string export_onnx(
    const std::string& target_path,
    const IrCaptureFunction& fun,
    const mlx::core::Args& args,
    const mlx::core::Kwargs& kwargs,
    bool shapeless,
    int64_t opset,
    const std::string& model_name,
    const OnnxBinaryWriteOptions& options);

MLX_API OnnxBinaryArtifact build_onnx_binary_artifact_from_onnx_json(
    const std::string& onnx_json,
    const OnnxBinaryWriteOptions& options);

MLX_API std::string write_onnx_binary_artifact_to_path(
    const std::string& target_path,
    const OnnxBinaryArtifact& artifact,
    const OnnxBinaryWriteOptions& options);

MLX_API bool ir_is_unsupported_error_message(const std::string& message);

} // namespace mlx::onnx

#include "detail.hpp"
#include "json.hpp"

#include <sstream>
#include <stdexcept>

namespace mlx::onnx {
namespace {

OrderedJson parse_json_payload_from_string(
    const std::string& raw,
    const char* label) {
  try {
    return OrderedJson::parse(raw);
  } catch (const std::exception& error) {
    std::ostringstream out;
    out << "failed to parse " << label << ": " << error.what();
    throw std::invalid_argument(
        detail::tagged_error_message("ir.api", out.str()));
  }
}

template <typename Result, typename Callable>
Result with_ir_api_error_tag(Callable&& callable) {
  try {
    return callable();
  } catch (const std::exception& error) {
    throw std::runtime_error(
        detail::tagged_error_message("ir.api", error.what()));
  }
}

std::string ir_to_onnx_json_impl(
    const std::string& ir_json,
    int64_t opset,
    const std::string& model_name) {
  const auto payload =
      parse_json_payload_from_string(ir_json, "graph ir json");
  return ir_to_onnx_json_payload(payload, opset, model_name).dump();
}

std::string ir_compatibility_report_json_impl(
    const std::string& ir_json) {
  const auto payload =
      parse_json_payload_from_string(ir_json, "graph ir json");
  return ir_compatibility_report_payload(payload).dump();
}

OnnxBinaryArtifact build_onnx_binary_artifact_from_onnx_json_impl(
    const std::string& onnx_json,
    const OnnxBinaryWriteOptions& options) {
  const auto payload = parse_json_payload_from_string(onnx_json, "onnx json");
  return build_onnx_binary_artifact_from_stub(payload, options);
}

} // namespace

std::string ir_to_onnx_json(
    const std::string& ir_json,
    int64_t opset,
    const std::string& model_name) {
  return with_ir_api_error_tag<std::string>([&]() {
    return ir_to_onnx_json_impl(ir_json, opset, model_name);
  });
}

std::string ir_compatibility_report_json(
    const std::string& ir_json) {
  return with_ir_api_error_tag<std::string>(
      [&]() { return ir_compatibility_report_json_impl(ir_json); });
}

OnnxBinaryArtifact build_onnx_binary_artifact_from_onnx_json(
    const std::string& onnx_json,
    const OnnxBinaryWriteOptions& options) {
  return with_ir_api_error_tag<OnnxBinaryArtifact>([&]() {
    return build_onnx_binary_artifact_from_onnx_json_impl(onnx_json, options);
  });
}

} // namespace mlx::onnx

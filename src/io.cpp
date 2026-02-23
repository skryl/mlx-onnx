#include "mlx/ir.hpp"
#include "detail.hpp"

#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "mlx/io/load.h"

namespace mlx::onnx {
namespace {

constexpr const char* kGraphIrIoTag = "ir.io";
constexpr const char* kGraphIrLoweringUnsupportedPrefix =
    "[ir.lowering] unsupported";
constexpr const char* kGraphIrLegacyUnsupportedPrefix =
    "[ir_to_onnx_stub] unsupported";

void write_binary_file(
    const std::filesystem::path& path,
    const std::string& bytes) {
  mlx::core::io::FileWriter output(path.string());
  if (!output.good()) {
    std::ostringstream out;
    out << "failed to open file for write: " << path.string();
    throw std::runtime_error(
        detail::tagged_error_message(kGraphIrIoTag, out.str()));
  }
  output.write(bytes.data(), bytes.size());
}

} // namespace

std::string write_onnx_binary_artifact_to_path(
    const std::string& target_path,
    const OnnxBinaryArtifact& artifact,
    const OnnxBinaryWriteOptions& options) {
  std::filesystem::path path(target_path);
  if (!path.has_parent_path()) {
    path = std::filesystem::absolute(path);
  }
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }

  write_binary_file(path, artifact.model_bytes);
  if (options.external_data && artifact.has_external_data) {
    write_binary_file(
        parent / options.external_data_file, artifact.external_data_bytes);
  }

  return path.string();
}

bool ir_is_unsupported_error_message(const std::string& message) {
  return message.find(kGraphIrLoweringUnsupportedPrefix) != std::string::npos ||
      message.find(kGraphIrLegacyUnsupportedPrefix) != std::string::npos;
}

} // namespace mlx::onnx

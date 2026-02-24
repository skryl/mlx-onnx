# C++ Interface

The standalone library exports the IR/ONNX API consumed by Python bindings
through `include/mlx/ir.hpp` under `mlx::onnx`:

- `export_ir_json`
- `export_onnx_compatibility_report_json`
- `export_onnx_json`
- `export_onnx`
- `ir_to_onnx_json`
- `ir_to_onnx`
- `ir_compatibility_report_json`
- `build_onnx_binary_artifact_from_onnx_json`
- `write_onnx_binary_artifact_to_path`
- `ir_is_unsupported_error_message`

`namespace`: `mlx::onnx` in `include/mlx/ir.hpp`

Shared types:

- `using IrCaptureFunction = std::function<std::vector<mlx::core::array>(const mlx::core::Args&, const mlx::core::Kwargs&)>;`
- `struct OnnxBinaryWriteOptions { bool external_data; std::string external_data_file; int64_t external_data_size_threshold; };`
  Defaults: `external_data=false`, `external_data_file="weights.bin"`, `external_data_size_threshold=1024`.
- `struct OnnxBinaryArtifact { std::string model_bytes; std::string external_data_bytes; bool has_external_data; };`

Function signatures and semantics:

- `std::string ir_to_onnx_json(const std::string& ir_json, int64_t opset, const std::string& model_name)`
  Input: IR JSON payload as string, ONNX opset (must be positive), model name (must be non-empty).
  Returns ONNX JSON string.

- `std::string ir_to_onnx(const std::string& target_path, const std::string& ir_json, int64_t opset, const std::string& model_name, const OnnxBinaryWriteOptions& options)`
  Same IR/metadata inputs as above plus output path and write options.
  Returns the written ONNX artifact path.

- `std::string ir_compatibility_report_json(const std::string& ir_json)`
  Input: IR JSON string.
  Returns compatibility report JSON string.

- `std::string export_ir_json(const IrCaptureFunction& fun, const mlx::core::Args& args, const mlx::core::Kwargs& kwargs, bool shapeless)`
  Inputs: captured function and concrete positional/keyword tracing inputs.
  `shapeless=true` allows shape-agnostic tracing behavior in supported paths.
  Returns traced MLX IR JSON string.

- `std::string export_onnx_compatibility_report_json(const IrCaptureFunction& fun, const mlx::core::Args& args, const mlx::core::Kwargs& kwargs, bool shapeless)`
  Same function/input contract as `export_ir_json`.
  Returns compatibility report JSON string.

- `std::string export_onnx_json(const IrCaptureFunction& fun, const mlx::core::Args& args, const mlx::core::Kwargs& kwargs, bool shapeless, int64_t opset, const std::string& model_name)`
  Traces and directly emits ONNX JSON.
  Returns ONNX JSON string.

- `std::string export_onnx(const std::string& target_path, const IrCaptureFunction& fun, const mlx::core::Args& args, const mlx::core::Kwargs& kwargs, bool shapeless, int64_t opset, const std::string& model_name, const OnnxBinaryWriteOptions& options)`
  Same tracing and conversion inputs as above plus output path and write options.
  Returns the written ONNX artifact path.

- `OnnxBinaryArtifact build_onnx_binary_artifact_from_onnx_json(const std::string& onnx_json, const OnnxBinaryWriteOptions& options)`
  Input: ONNX JSON text and write options.
  Returns `OnnxBinaryArtifact` containing serialized model bytes and optional external tensor bytes.

- `std::string write_onnx_binary_artifact_to_path(const std::string& target_path, const OnnxBinaryArtifact& artifact, const OnnxBinaryWriteOptions& options)`
  Writes a binary ONNX artifact plus optional external tensor file(s) as configured.
  Returns final written ONNX path.

- `bool ir_is_unsupported_error_message(const std::string& message)`
  Returns `true` when `message` represents an unsupported-operator IR export error.

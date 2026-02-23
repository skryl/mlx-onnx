#include "mlx/ir.hpp"

#include "detail.hpp"
#include "json.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "mlx/ops.h"

namespace mx = mlx::core;

namespace mlx::onnx {
namespace {

using GraphTensorInfo = std::tuple<std::string, mx::Shape, mx::Dtype>;

std::string dtype_to_string(mx::Dtype dtype) {
  std::ostringstream out;
  out << dtype;
  return out.str();
}

template <typename ValueAt>
OrderedJson capture_build_nested_json_array(
    const mx::Shape& shape,
    size_t dim,
    size_t& flat_index,
    ValueAt value_at) {
  if (dim == shape.size()) {
    return value_at(flat_index++);
  }

  OrderedJson out = OrderedJson::array();
  for (size_t i = 0; i < shape[dim]; ++i) {
    out.push_back(
        capture_build_nested_json_array(shape, dim + 1, flat_index, value_at));
  }
  return out;
}

template <typename ValueAt>
OrderedJson capture_build_flat_json_array(size_t size, ValueAt value_at) {
  OrderedJson out = OrderedJson::array();
  for (size_t i = 0; i < size; ++i) {
    out.push_back(value_at(i));
  }
  return out;
}

OrderedJson capture_json_shape_from_mx_shape(const mx::Shape& shape) {
  OrderedJson out = OrderedJson::array();
  for (size_t dim : shape) {
    out.push_back(dim);
  }
  return out;
}

OrderedJson capture_json_scalar_from_array(const mx::array& array) {
  switch (array.dtype()) {
    case mx::bool_:
      return OrderedJson(array.item<bool>());
    case mx::uint8:
      return OrderedJson(array.item<uint8_t>());
    case mx::uint16:
      return OrderedJson(array.item<uint16_t>());
    case mx::uint32:
      return OrderedJson(array.item<uint32_t>());
    case mx::uint64:
      return OrderedJson(array.item<uint64_t>());
    case mx::int8:
      return OrderedJson(array.item<int8_t>());
    case mx::int16:
      return OrderedJson(array.item<int16_t>());
    case mx::int32:
      return OrderedJson(array.item<int32_t>());
    case mx::int64:
      return OrderedJson(array.item<int64_t>());
    case mx::float16:
      return OrderedJson(static_cast<double>(array.item<mx::float16_t>()));
    case mx::bfloat16:
      return OrderedJson(static_cast<double>(array.item<mx::bfloat16_t>()));
    case mx::float32:
      return OrderedJson(static_cast<double>(array.item<float>()));
    case mx::float64:
      return OrderedJson(array.item<double>());
    default:
      throw std::runtime_error("unsupported dtype for graph ir constant conversion");
  }
}

OrderedJson capture_json_values_from_array(const mx::array& source) {
  mx::array array = source;
  if (array.ndim() == 0) {
    array.eval();
    return capture_json_scalar_from_array(array);
  }

  if (array.ndim() == 1) {
    array.eval();
    const size_t size = array.size();
    switch (array.dtype()) {
      case mx::bool_: {
        const bool* data = array.data<bool>();
        return capture_build_flat_json_array(
            size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::uint8: {
        const uint8_t* data = array.data<uint8_t>();
        return capture_build_flat_json_array(
            size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::uint16: {
        const uint16_t* data = array.data<uint16_t>();
        return capture_build_flat_json_array(
            size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::uint32: {
        const uint32_t* data = array.data<uint32_t>();
        return capture_build_flat_json_array(
            size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::uint64: {
        const uint64_t* data = array.data<uint64_t>();
        return capture_build_flat_json_array(
            size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::int8: {
        const int8_t* data = array.data<int8_t>();
        return capture_build_flat_json_array(
            size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::int16: {
        const int16_t* data = array.data<int16_t>();
        return capture_build_flat_json_array(
            size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::int32: {
        const int32_t* data = array.data<int32_t>();
        return capture_build_flat_json_array(
            size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::int64: {
        const int64_t* data = array.data<int64_t>();
        return capture_build_flat_json_array(
            size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::float16: {
        const mx::float16_t* data = array.data<mx::float16_t>();
        return capture_build_flat_json_array(
            size,
            [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
      }
      case mx::bfloat16: {
        const mx::bfloat16_t* data = array.data<mx::bfloat16_t>();
        return capture_build_flat_json_array(
            size,
            [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
      }
      case mx::float32: {
        const float* data = array.data<float>();
        return capture_build_flat_json_array(
            size,
            [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
      }
      case mx::float64: {
        const double* data = array.data<double>();
        return capture_build_flat_json_array(
            size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      default:
        throw std::runtime_error(
            "unsupported dtype for graph ir constant conversion");
    }
  }

  const mx::Shape shape = array.shape();
  mx::array flat =
      mx::reshape(array, mx::Shape{static_cast<mx::ShapeElem>(array.size())});
  flat.eval();

  size_t idx = 0;
  switch (flat.dtype()) {
    case mx::bool_: {
      const bool* data = flat.data<bool>();
      return capture_build_nested_json_array(
          shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::uint8: {
      const uint8_t* data = flat.data<uint8_t>();
      return capture_build_nested_json_array(
          shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::uint16: {
      const uint16_t* data = flat.data<uint16_t>();
      return capture_build_nested_json_array(
          shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::uint32: {
      const uint32_t* data = flat.data<uint32_t>();
      return capture_build_nested_json_array(
          shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::uint64: {
      const uint64_t* data = flat.data<uint64_t>();
      return capture_build_nested_json_array(
          shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::int8: {
      const int8_t* data = flat.data<int8_t>();
      return capture_build_nested_json_array(
          shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::int16: {
      const int16_t* data = flat.data<int16_t>();
      return capture_build_nested_json_array(
          shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::int32: {
      const int32_t* data = flat.data<int32_t>();
      return capture_build_nested_json_array(
          shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::int64: {
      const int64_t* data = flat.data<int64_t>();
      return capture_build_nested_json_array(
          shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::float16: {
      const mx::float16_t* data = flat.data<mx::float16_t>();
      return capture_build_nested_json_array(
          shape,
          0,
          idx,
          [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
    }
    case mx::bfloat16: {
      const mx::bfloat16_t* data = flat.data<mx::bfloat16_t>();
      return capture_build_nested_json_array(
          shape,
          0,
          idx,
          [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
    }
    case mx::float32: {
      const float* data = flat.data<float>();
      return capture_build_nested_json_array(
          shape,
          0,
          idx,
          [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
    }
    case mx::float64: {
      const double* data = flat.data<double>();
      return capture_build_nested_json_array(
          shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    default:
      throw std::runtime_error(
          "unsupported dtype for graph ir constant conversion");
  }
}

OrderedJson capture_json_tensor_info_from_graph_tensor(const GraphTensorInfo& info) {
  OrderedJson out = OrderedJson::object();
  out["name"] = std::get<0>(info);
  out["shape"] = capture_json_shape_from_mx_shape(std::get<1>(info));
  out["dtype"] = dtype_to_string(std::get<2>(info));
  return out;
}

OrderedJson capture_json_tensor_infos_from_graph_tensors(
    const std::vector<GraphTensorInfo>& infos) {
  OrderedJson out = OrderedJson::array();
  for (const auto& info : infos) {
    out.push_back(capture_json_tensor_info_from_graph_tensor(info));
  }
  return out;
}

OrderedJson capture_json_tensor_names_from_graph_tensors(
    const std::vector<GraphTensorInfo>& infos) {
  OrderedJson out = OrderedJson::array();
  for (const auto& info : infos) {
    out.push_back(std::get<0>(info));
  }
  return out;
}

OrderedJson capture_json_state_value_from_mx_state(const mx::StateT& value) {
  if (std::holds_alternative<bool>(value)) {
    return OrderedJson(std::get<bool>(value));
  }
  if (std::holds_alternative<int>(value)) {
    return OrderedJson(std::get<int>(value));
  }
  if (std::holds_alternative<size_t>(value)) {
    return OrderedJson(std::get<size_t>(value));
  }
  if (std::holds_alternative<float>(value)) {
    return OrderedJson(static_cast<double>(std::get<float>(value)));
  }
  if (std::holds_alternative<double>(value)) {
    return OrderedJson(std::get<double>(value));
  }
  if (std::holds_alternative<mx::Dtype>(value)) {
    return OrderedJson(dtype_to_string(std::get<mx::Dtype>(value)));
  }
  if (std::holds_alternative<mx::Shape>(value)) {
    return capture_json_shape_from_mx_shape(std::get<mx::Shape>(value));
  }
  if (std::holds_alternative<mx::Strides>(value)) {
    OrderedJson out = OrderedJson::array();
    const auto& strides = std::get<mx::Strides>(value);
    for (auto stride : strides) {
      out.push_back(static_cast<long long>(stride));
    }
    return out;
  }
  if (std::holds_alternative<std::vector<int>>(value)) {
    OrderedJson out = OrderedJson::array();
    const auto& values = std::get<std::vector<int>>(value);
    for (int item : values) {
      out.push_back(item);
    }
    return out;
  }
  if (std::holds_alternative<std::vector<size_t>>(value)) {
    OrderedJson out = OrderedJson::array();
    const auto& values = std::get<std::vector<size_t>>(value);
    for (size_t item : values) {
      out.push_back(item);
    }
    return out;
  }
  if (std::holds_alternative<std::vector<std::tuple<bool, bool, bool>>>(value)) {
    OrderedJson out = OrderedJson::array();
    const auto& tuples = std::get<std::vector<std::tuple<bool, bool, bool>>>(value);
    for (const auto& item : tuples) {
      out.push_back(
          OrderedJson::array({std::get<0>(item), std::get<1>(item), std::get<2>(item)}));
    }
    return out;
  }
  if (std::holds_alternative<std::vector<std::variant<bool, int, float>>>(value)) {
    OrderedJson out = OrderedJson::array();
    const auto& vars = std::get<std::vector<std::variant<bool, int, float>>>(value);
    for (const auto& item : vars) {
      if (std::holds_alternative<bool>(item)) {
        out.push_back(std::get<bool>(item));
      } else if (std::holds_alternative<int>(item)) {
        out.push_back(std::get<int>(item));
      } else {
        out.push_back(static_cast<double>(std::get<float>(item)));
      }
    }
    return out;
  }
  if (std::holds_alternative<std::optional<float>>(value)) {
    const auto& opt = std::get<std::optional<float>>(value);
    if (!opt.has_value()) {
      return nullptr;
    }
    return OrderedJson(static_cast<double>(opt.value()));
  }
  return OrderedJson(std::get<std::string>(value));
}

OrderedJson capture_json_state_values_from_mx_states(
    const std::vector<mx::StateT>& values) {
  OrderedJson out = OrderedJson::array();
  for (const auto& value : values) {
    out.push_back(capture_json_state_value_from_mx_state(value));
  }
  return out;
}

template <typename T>
const T* export_callback_field(
    const mx::ExportCallbackInput& data,
    const std::string& key) {
  for (const auto& [candidate_key, candidate_value] : data) {
    if (candidate_key == key && std::holds_alternative<T>(candidate_value)) {
      return &std::get<T>(candidate_value);
    }
  }
  return nullptr;
}

OrderedJson export_ir_payload(
    const IrCaptureFunction& fun,
    const mx::Args& args,
    const mx::Kwargs& kwargs,
    bool shapeless) {
  OrderedJson graph_inputs = OrderedJson::array();
  OrderedJson keyword_inputs = OrderedJson::array();
  OrderedJson graph_outputs = OrderedJson::array();
  OrderedJson graph_constants = OrderedJson::array();
  OrderedJson graph_nodes = OrderedJson::array();

  mx::export_function(
      [&](const mx::ExportCallbackInput& data) {
        const auto* record_type = export_callback_field<std::string>(data, "type");
        if (record_type == nullptr) {
          return;
        }

        if (*record_type == "inputs") {
          const auto* inputs =
              export_callback_field<std::vector<GraphTensorInfo>>(data, "inputs");
          if (inputs != nullptr) {
            graph_inputs = capture_json_tensor_infos_from_graph_tensors(*inputs);
          }
          return;
        }

        if (*record_type == "keyword_inputs") {
          const auto* keywords =
              export_callback_field<std::vector<std::pair<std::string, std::string>>>(
                  data,
                  "keywords");
          if (keywords != nullptr) {
            keyword_inputs = OrderedJson::array();
            for (const auto& [name, tensor] : *keywords) {
              OrderedJson entry = OrderedJson::object();
              entry["name"] = name;
              entry["tensor"] = tensor;
              keyword_inputs.push_back(std::move(entry));
            }
          }
          return;
        }

        if (*record_type == "outputs") {
          const auto* outputs =
              export_callback_field<std::vector<GraphTensorInfo>>(data, "outputs");
          if (outputs != nullptr) {
            graph_outputs = capture_json_tensor_infos_from_graph_tensors(*outputs);
          }
          return;
        }

        if (*record_type == "constants") {
          const auto* constants =
              export_callback_field<std::vector<std::pair<std::string, mx::array>>>(
                  data,
                  "constants");
          if (constants != nullptr) {
            graph_constants = OrderedJson::array();
            for (const auto& [name, arr] : *constants) {
              OrderedJson entry = OrderedJson::object();
              entry["name"] = name;
              entry["shape"] = capture_json_shape_from_mx_shape(arr.shape());
              entry["dtype"] = dtype_to_string(arr.dtype());
              entry["values"] = capture_json_values_from_array(arr);
              graph_constants.push_back(std::move(entry));
            }
          }
          return;
        }

        if (*record_type != "primitive") {
          return;
        }

        const auto* op_name = export_callback_field<std::string>(data, "name");
        if (op_name == nullptr) {
          return;
        }

        OrderedJson node = OrderedJson::object();
        node["op"] = *op_name;

        OrderedJson node_inputs = OrderedJson::array();
        const auto* node_input_infos =
            export_callback_field<std::vector<GraphTensorInfo>>(data, "inputs");
        if (node_input_infos != nullptr) {
          node_inputs = capture_json_tensor_names_from_graph_tensors(*node_input_infos);
        }
        node["inputs"] = std::move(node_inputs);

        OrderedJson node_outputs = OrderedJson::array();
        const auto* node_output_infos =
            export_callback_field<std::vector<GraphTensorInfo>>(data, "outputs");
        if (node_output_infos != nullptr) {
          node_outputs = capture_json_tensor_names_from_graph_tensors(*node_output_infos);
        }
        node["outputs"] = std::move(node_outputs);

        OrderedJson node_arguments = OrderedJson::array();
        const auto* arguments =
            export_callback_field<std::vector<mx::StateT>>(data, "arguments");
        if (arguments != nullptr) {
          node_arguments = capture_json_state_values_from_mx_states(*arguments);
        }
        node["arguments"] = std::move(node_arguments);

        graph_nodes.push_back(std::move(node));
      },
      fun,
      args,
      kwargs,
      shapeless);

  OrderedJson payload = OrderedJson::object();
  payload["ir_version"] = kGraphIrVersion;
  payload["shapeless"] = shapeless;
  payload["inputs"] = std::move(graph_inputs);
  payload["keyword_inputs"] = std::move(keyword_inputs);
  payload["outputs"] = std::move(graph_outputs);
  payload["constants"] = std::move(graph_constants);
  payload["nodes"] = std::move(graph_nodes);
  return payload;
}

int64_t normalize_positive_integer(int64_t value, const char* label) {
  if (value <= 0) {
    std::ostringstream out;
    out << label << " must be a positive integer";
    throw std::invalid_argument(detail::tagged_error_message("ir.api", out.str()));
  }
  return value;
}

std::string non_empty_model_name(std::string value) {
  if (value.empty()) {
    throw std::invalid_argument(
        detail::tagged_error_message("ir.api", "model_name must not be empty"));
  }
  return value;
}

void validate_onnx_binary_write_options(const OnnxBinaryWriteOptions& options) {
  if (!options.external_data) {
    return;
  }

  if (options.external_data_size_threshold < 0) {
    throw std::invalid_argument(
        detail::tagged_error_message(
            "ir.api", "external_data_size_threshold must be non-negative"));
  }

  if (options.external_data_file.empty()) {
    throw std::invalid_argument(
        detail::tagged_error_message(
            "ir.api", "external_data_file must not be empty when provided"));
  }
}

template <typename Result, typename Callable>
Result with_ir_api_error_tag(Callable&& callable) {
  try {
    return callable();
  } catch (const std::exception& error) {
    if (ir_is_unsupported_error_message(error.what())) {
      throw;
    }
    throw std::runtime_error(
        detail::tagged_error_message("ir.api", error.what()));
  }
}

} // namespace

std::string export_ir_json(
    const IrCaptureFunction& fun,
    const mx::Args& args,
    const mx::Kwargs& kwargs,
    bool shapeless) {
  return with_ir_api_error_tag<std::string>(
      [&]() { return export_ir_payload(fun, args, kwargs, shapeless).dump(); });
}

std::string export_onnx_compatibility_report_json(
    const IrCaptureFunction& fun,
    const mx::Args& args,
    const mx::Kwargs& kwargs,
    bool shapeless) {
  return with_ir_api_error_tag<std::string>([&]() {
    const auto payload = export_ir_payload(fun, args, kwargs, shapeless);
    return ir_compatibility_report_payload(payload).dump();
  });
}

std::string export_onnx_json(
    const IrCaptureFunction& fun,
    const mx::Args& args,
    const mx::Kwargs& kwargs,
    bool shapeless,
    int64_t opset,
    const std::string& model_name) {
  return with_ir_api_error_tag<std::string>([&]() {
    const auto payload = export_ir_payload(fun, args, kwargs, shapeless);
    const auto onnx = ir_to_onnx_json_payload(
        payload,
        normalize_positive_integer(opset, "opset"),
        non_empty_model_name(model_name));
    return onnx.dump();
  });
}

std::string export_onnx(
    const std::string& target_path,
    const IrCaptureFunction& fun,
    const mx::Args& args,
    const mx::Kwargs& kwargs,
    bool shapeless,
    int64_t opset,
    const std::string& model_name,
    const OnnxBinaryWriteOptions& options) {
  return with_ir_api_error_tag<std::string>([&]() {
    validate_onnx_binary_write_options(options);
    const auto onnx_json = export_onnx_json(
        fun,
        args,
        kwargs,
        shapeless,
        normalize_positive_integer(opset, "opset"),
        non_empty_model_name(model_name));
    const auto artifact =
        build_onnx_binary_artifact_from_onnx_json(onnx_json, options);
    return write_onnx_binary_artifact_to_path(target_path, artifact, options);
  });
}

std::string ir_to_onnx(
    const std::string& target_path,
    const std::string& ir_json,
    int64_t opset,
    const std::string& model_name,
    const OnnxBinaryWriteOptions& options) {
  return with_ir_api_error_tag<std::string>([&]() {
    validate_onnx_binary_write_options(options);
    const auto onnx_json = ir_to_onnx_json(
        ir_json,
        normalize_positive_integer(opset, "opset"),
        non_empty_model_name(model_name));
    const auto artifact =
        build_onnx_binary_artifact_from_onnx_json(onnx_json, options);
    return write_onnx_binary_artifact_to_path(target_path, artifact, options);
  });
}

} // namespace mlx::onnx

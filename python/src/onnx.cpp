#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "mlx/array.h"
#include "mlx/export.h"
#include "mlx/ops.h"
#include "mlx/ir.hpp"
#include "json.hpp"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;
using OrderedJson = nlohmann::ordered_json;

namespace {

using GraphTensorInfo = std::tuple<std::string, mx::Shape, mx::Dtype>;

std::pair<mx::Args, mx::Kwargs> validate_and_extract_trace_inputs(
    const nb::args& args,
    const nb::kwargs& kwargs,
    const std::string& prefix) {
  auto maybe_throw = [&prefix](bool valid) {
    if (!valid) {
      throw std::invalid_argument(
          prefix +
          " Inputs can either be a variable "
          "number of positional and keyword arrays or a single tuple "
          "and/or dictionary of arrays.");
    }
  };

  mx::Args args_;
  mx::Kwargs kwargs_;

  if (args.size() == 0) {
    maybe_throw(nb::try_cast(kwargs, kwargs_));
  } else if (args.size() > 0 && nb::isinstance<mx::array>(args[0])) {
    maybe_throw(nb::try_cast(args, args_));
    maybe_throw(nb::try_cast(kwargs, kwargs_));
  } else if (args.size() == 1) {
    maybe_throw(kwargs.size() == 0);
    if (!nb::try_cast(args[0], args_)) {
      maybe_throw(nb::try_cast(args[0], kwargs_));
    }
  } else if (args.size() == 2) {
    maybe_throw(kwargs.size() == 0);
    maybe_throw(nb::try_cast(args[0], args_));
    maybe_throw(nb::try_cast(args[1], kwargs_));
  } else {
    maybe_throw(false);
  }

  return {args_, kwargs_};
}

auto wrap_ir_callable(nb::callable fun) {
  return [fun = std::move(fun)](const mx::Args& args_, const mx::Kwargs& kwargs_) {
    nb::gil_scoped_acquire gil;

    auto kwargs = nb::dict();
    kwargs.update(nb::cast(kwargs_));

    auto args = nb::tuple(nb::cast(args_));
    auto outputs = fun(*args, **kwargs);

    std::vector<mx::array> outputs_;
    if (nb::isinstance<mx::array>(outputs)) {
      outputs_.push_back(nb::cast<mx::array>(outputs));
    } else if (!nb::try_cast(outputs, outputs_)) {
      throw std::invalid_argument(
          "[ir] Outputs can be either a single array "
          "or a tuple/list of arrays.");
    }

    return outputs_;
  };
}

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
      throw std::runtime_error(
          "unsupported dtype for graph ir constant conversion");
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
    const std::function<std::vector<mx::array>(const mx::Args&, const mx::Kwargs&)>&
        fun,
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
  payload["ir_version"] = mlx::onnx::kGraphIrVersion;
  payload["shapeless"] = shapeless;
  payload["inputs"] = std::move(graph_inputs);
  payload["keyword_inputs"] = std::move(keyword_inputs);
  payload["outputs"] = std::move(graph_outputs);
  payload["constants"] = std::move(graph_constants);
  payload["nodes"] = std::move(graph_nodes);
  return payload;
}

OrderedJson ordered_json_integer_from_python(nb::handle value) {
  const std::string text = nb::cast<std::string>(nb::str(value));
  if (text.empty()) {
    throw std::invalid_argument(
        "[ir.api] failed to convert Integer to JSON number");
  }

  const bool negative = text.front() == '-';
  try {
    if (negative) {
      return static_cast<int64_t>(std::stoll(text));
    }

    const auto raw = std::stoull(text);
    if (raw <= static_cast<unsigned long long>(std::numeric_limits<int64_t>::max())) {
      return static_cast<int64_t>(raw);
    }
    return static_cast<uint64_t>(raw);
  } catch (const std::out_of_range&) {
    try {
      return static_cast<double>(std::stold(text));
    } catch (const std::exception&) {
      throw std::invalid_argument(
          "[ir.api] Integer is too large to convert into JSON numeric range");
    }
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument(
        "[ir.api] failed to parse Integer while converting to JSON");
  }
}

OrderedJson ordered_json_from_python(nb::handle value);

OrderedJson ordered_json_array_from_python(nb::handle value) {
  OrderedJson out = OrderedJson::array();

  if (nb::isinstance<nb::list>(value)) {
    nb::list list = nb::borrow<nb::list>(value);
    for (nb::handle item : list) {
      out.push_back(ordered_json_from_python(item));
    }
    return out;
  }

  nb::tuple tuple = nb::borrow<nb::tuple>(value);
  for (nb::handle item : tuple) {
    out.push_back(ordered_json_from_python(item));
  }
  return out;
}

OrderedJson ordered_json_object_from_python_dict(nb::handle value) {
  OrderedJson out = OrderedJson::object();
  nb::dict dict = nb::borrow<nb::dict>(value);
  for (auto item : dict) {
    const std::string key = nb::cast<std::string>(nb::str(item.first));
    out[key] = ordered_json_from_python(item.second);
  }
  return out;
}

OrderedJson ordered_json_from_python(nb::handle value) {
  if (value.is_none()) {
    return nullptr;
  }

  if (PyBool_Check(value.ptr())) {
    return OrderedJson(nb::cast<bool>(value));
  }

  if (PyLong_Check(value.ptr())) {
    return ordered_json_integer_from_python(value);
  }

  if (PyFloat_Check(value.ptr())) {
    return OrderedJson(nb::cast<double>(value));
  }

  if (nb::isinstance<nb::str>(value)) {
    return OrderedJson(nb::cast<std::string>(value));
  }

  if (nb::isinstance<nb::dict>(value)) {
    return ordered_json_object_from_python_dict(value);
  }

  if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
    return ordered_json_array_from_python(value);
  }

  throw std::invalid_argument(
      "[ir.api] unsupported graph ir value; expected JSON-compatible value");
}

nb::object python_value_from_ordered_json(const OrderedJson& value) {
  if (value.is_null()) {
    return nb::none();
  }
  if (value.is_boolean()) {
    return nb::cast(value.get<bool>());
  }
  if (value.is_number_integer()) {
    return nb::cast(value.get<int64_t>());
  }
  if (value.is_number_unsigned()) {
    return nb::cast(value.get<uint64_t>());
  }
  if (value.is_number_float()) {
    return nb::cast(value.get<double>());
  }
  if (value.is_string()) {
    return nb::cast(value.get<std::string>());
  }
  if (value.is_array()) {
    nb::list out;
    for (const auto& item : value) {
      out.append(python_value_from_ordered_json(item));
    }
    return std::move(out);
  }
  if (value.is_object()) {
    nb::dict out;
    for (auto it = value.begin(); it != value.end(); ++it) {
      out[nb::str(it.key().c_str())] = python_value_from_ordered_json(it.value());
    }
    return std::move(out);
  }

  throw std::runtime_error("unsupported ordered_json value type");
}

std::string read_file_to_string(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    std::ostringstream out;
    out << "[ir.api] failed to read file: " << path;
    throw std::runtime_error(out.str());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

OrderedJson parse_json_payload_from_string(
    const std::string& raw,
    const std::string& label) {
  try {
    return OrderedJson::parse(raw);
  } catch (const std::exception& error) {
    std::ostringstream out;
    out << "[ir.api] failed to parse " << label << ": " << error.what();
    throw std::invalid_argument(out.str());
  }
}

std::string python_path_string(nb::handle value, const char* label) {
  if (nb::hasattr(value, "write") && !nb::hasattr(value, "__fspath__") &&
      !nb::isinstance<nb::str>(value)) {
    std::ostringstream out;
    out << "[ir.api] " << label
        << " requires a path-like target, not an IO-like target";
    throw std::invalid_argument(out.str());
  }

  if (nb::hasattr(value, "__fspath__")) {
    nb::object fs_path = value.attr("__fspath__")();
    const std::string out = nb::cast<std::string>(fs_path);
    if (out.empty()) {
      std::ostringstream msg;
      msg << "[ir.api] " << label
          << " target must be a non-empty path-like value";
      throw std::invalid_argument(msg.str());
    }
    return out;
  }

  if (nb::isinstance<nb::str>(value)) {
    const std::string out = nb::cast<std::string>(value);
    if (out.empty()) {
      std::ostringstream msg;
      msg << "[ir.api] " << label
          << " target must be a non-empty path-like value";
      throw std::invalid_argument(msg.str());
    }
    return out;
  }

  std::ostringstream msg;
  msg << "[ir.api] " << label
      << " target must be a String or path-like value";
  throw std::invalid_argument(msg.str());
}

OrderedJson parse_ir_source_payload(nb::handle source) {
  if (nb::isinstance<nb::dict>(source) || nb::isinstance<nb::list>(source) ||
      nb::isinstance<nb::tuple>(source)) {
    return ordered_json_from_python(source);
  }

  if (nb::isinstance<nb::str>(source)) {
    const std::string raw = nb::cast<std::string>(source);

    bool treat_as_file = false;
    try {
      treat_as_file = std::filesystem::is_regular_file(raw);
    } catch (const std::filesystem::filesystem_error&) {
      treat_as_file = false;
    }

    if (treat_as_file) {
      return parse_json_payload_from_string(
          read_file_to_string(raw), "graph ir file");
    }

    return parse_json_payload_from_string(raw, "graph ir string");
  }

  if (nb::hasattr(source, "__fspath__")) {
    const std::string path = python_path_string(source, "ir_source");
    if (!std::filesystem::is_regular_file(path)) {
      std::ostringstream out;
      out << "[ir.api] graph ir path does not exist: " << path;
      throw std::invalid_argument(out.str());
    }
    return parse_json_payload_from_string(read_file_to_string(path), "graph ir file");
  }

  if (nb::hasattr(source, "read")) {
    nb::object raw = source.attr("read")();
    return parse_json_payload_from_string(
        nb::cast<std::string>(raw), "graph ir IO");
  }

  throw std::invalid_argument(
      "[ir.api] graph ir source must be a dict/list/tuple, JSON string, "
      "file path, or IO-like object");
}

int64_t normalize_positive_integer(int64_t value, const char* label) {
  if (value <= 0) {
    std::ostringstream out;
    out << "[ir.api] " << label << " must be a positive integer";
    throw std::invalid_argument(out.str());
  }
  return value;
}

std::string non_empty_model_name(std::string value) {
  if (value.empty()) {
    throw std::invalid_argument("[ir.api] model_name must not be empty");
  }
  return value;
}

mlx::onnx::OnnxBinaryWriteOptions normalize_onnx_binary_write_options(
    const std::string& target_path,
    bool external_data,
    const std::optional<std::string>& external_data_file,
    int64_t external_data_size_threshold) {
  mlx::onnx::OnnxBinaryWriteOptions options;
  options.external_data = external_data;
  options.external_data_size_threshold = 1024;
  options.external_data_file = "weights.bin";

  if (!external_data) {
    return options;
  }

  if (external_data_size_threshold < 0) {
    throw std::invalid_argument(
        "[ir.api] external_data_size_threshold must be non-negative");
  }
  options.external_data_size_threshold = external_data_size_threshold;

  if (external_data_file.has_value()) {
    if (external_data_file->empty()) {
      throw std::invalid_argument(
          "[ir.api] external_data_file must not be empty when provided");
    }
    options.external_data_file = *external_data_file;
    return options;
  }

  std::filesystem::path path(target_path);
  std::string base = path.stem().string();
  if (base.empty()) {
    base = path.filename().string();
  }
  if (base.empty()) {
    base = "model";
  }
  options.external_data_file = base + ".data";
  return options;
}

template <typename T, typename Callable>
T with_ir_exception_translation(Callable&& callable) {
  try {
    return callable();
  } catch (const std::exception& error) {
    const std::string message = error.what();
    if (mlx::onnx::ir_is_unsupported_error_message(message)) {
      PyErr_SetString(PyExc_NotImplementedError, message.c_str());
      throw nb::python_error();
    }
    throw;
  }
}

std::string write_onnx_binary_from_payload(
    const std::string& target_path,
    const OrderedJson& payload,
    int64_t opset,
    const std::string& model_name,
    const mlx::onnx::OnnxBinaryWriteOptions& options) {
  const auto onnx_payload =
      mlx::onnx::ir_to_onnx_json_payload(payload, opset, model_name);
  const auto artifact =
      mlx::onnx::build_onnx_binary_artifact_from_stub(onnx_payload, options);
  return mlx::onnx::write_onnx_binary_artifact_to_path(
      target_path, artifact, options);
}

} // namespace

void init_onnx(nb::module_& m) {
  m.def(
      "export_ir",
      [](const nb::callable& fun,
         const nb::args& args,
         bool shapeless,
         const nb::kwargs& kwargs) {
        return with_ir_exception_translation<nb::object>([&]() {
          auto [args_, kwargs_] = validate_and_extract_trace_inputs(
              args, kwargs, "[export_ir]");
          const auto payload_json = mlx::onnx::export_ir_json(
              wrap_ir_callable(fun), args_, kwargs_, shapeless);
          const auto payload = OrderedJson::parse(payload_json);
          return python_value_from_ordered_json(payload);
        });
      },
      "fun"_a,
      "args"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      "kwargs"_a,
      nb::sig(
          "def export_ir(fun: Callable, *args, shapeless: bool = False, **kwargs) -> dict"));

  m.def(
      "export_ir_json",
      [](const nb::callable& fun,
         const nb::args& args,
         bool shapeless,
         const nb::kwargs& kwargs) {
        return with_ir_exception_translation<std::string>([&]() {
          auto [args_, kwargs_] = validate_and_extract_trace_inputs(
              args, kwargs, "[export_ir_json]");
          return mlx::onnx::export_ir_json(
              wrap_ir_callable(fun), args_, kwargs_, shapeless);
        });
      },
      "fun"_a,
      "args"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      "kwargs"_a,
      nb::sig(
          "def export_ir_json(fun: Callable, *args, shapeless: bool = False, **kwargs) -> str"));

  m.def(
      "export_onnx_compatibility_report",
      [](const nb::callable& fun,
         const nb::args& args,
         bool shapeless,
         const nb::kwargs& kwargs) {
        return with_ir_exception_translation<nb::object>([&]() {
          auto [args_, kwargs_] = validate_and_extract_trace_inputs(
              args, kwargs, "[export_onnx_compatibility_report]");
          const auto report_json =
              mlx::onnx::export_onnx_compatibility_report_json(
                  wrap_ir_callable(fun), args_, kwargs_, shapeless);
          const auto report = OrderedJson::parse(report_json);
          return python_value_from_ordered_json(report);
        });
      },
      "fun"_a,
      "args"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      "kwargs"_a,
      nb::sig(
          "def export_onnx_compatibility_report(fun: Callable, *args, shapeless: bool = False, **kwargs) -> dict"));

  m.def(
      "export_onnx_json",
      [](const nb::callable& fun,
         const nb::args& args,
         bool shapeless,
         int64_t opset,
         std::string model_name,
         const nb::kwargs& kwargs) {
        return with_ir_exception_translation<std::string>([&]() {
          auto [args_, kwargs_] = validate_and_extract_trace_inputs(
              args, kwargs, "[export_onnx_json]");
          return mlx::onnx::export_onnx_json(
              wrap_ir_callable(fun),
              args_,
              kwargs_,
              shapeless,
              normalize_positive_integer(opset, "opset"),
              non_empty_model_name(std::move(model_name)));
        });
      },
      "fun"_a,
      "args"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      "opset"_a = 18,
      "model_name"_a = std::string("mlx_graph"),
      "kwargs"_a,
      nb::sig(
          "def export_onnx_json(fun: Callable, *args, shapeless: bool = False, opset: int = 18, model_name: str = 'mlx_graph', **kwargs) -> str"));

  m.def(
      "export_onnx",
      [](nb::handle target_path,
         const nb::callable& fun,
         const nb::args& args,
         bool shapeless,
         int64_t opset,
         std::string model_name,
         bool external_data,
         std::optional<std::string> external_data_file,
         int64_t external_data_size_threshold,
         const nb::kwargs& kwargs) {
        return with_ir_exception_translation<std::string>([&]() {
          auto [args_, kwargs_] =
              validate_and_extract_trace_inputs(args, kwargs, "[export_onnx]");

          const auto target = python_path_string(target_path, "export_onnx");
          const auto options = normalize_onnx_binary_write_options(
              target,
              external_data,
              external_data_file,
              external_data_size_threshold);

          return mlx::onnx::export_onnx(
              target,
              wrap_ir_callable(fun),
              args_,
              kwargs_,
              shapeless,
              normalize_positive_integer(opset, "opset"),
              non_empty_model_name(std::move(model_name)),
              options);
        });
      },
      "target_path"_a,
      "fun"_a,
      "args"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      "opset"_a = 18,
      "model_name"_a = std::string("mlx_graph"),
      "external_data"_a = false,
      "external_data_file"_a = nb::none(),
      "external_data_size_threshold"_a = 1024,
      "kwargs"_a,
      nb::sig(
          "def export_onnx(target_path: Union[str, PathLike], fun: Callable, *args, shapeless: bool = False, opset: int = 18, model_name: str = 'mlx_graph', external_data: bool = False, external_data_file: Optional[str] = None, external_data_size_threshold: int = 1024, **kwargs) -> str"));

  m.def(
      "ir_to_onnx_json",
      [](nb::handle ir_source, int64_t opset, std::string model_name) {
        return with_ir_exception_translation<std::string>([&]() {
          const auto payload = parse_ir_source_payload(ir_source);
          return mlx::onnx::ir_to_onnx_json(
              payload.dump(),
              normalize_positive_integer(opset, "opset"),
              non_empty_model_name(std::move(model_name)));
        });
      },
      "ir_source"_a,
      nb::kw_only(),
      "opset"_a = 18,
      "model_name"_a = std::string("mlx_graph"),
      nb::sig(
          "def ir_to_onnx_json(ir_source: Union[dict, list, tuple, str, PathLike], *, opset: int = 18, model_name: str = 'mlx_graph') -> str"));

  m.def(
      "ir_to_onnx",
      [](nb::handle target_path,
         nb::handle ir_source,
         int64_t opset,
         std::string model_name,
         bool external_data,
         std::optional<std::string> external_data_file,
         int64_t external_data_size_threshold) {
        return with_ir_exception_translation<std::string>([&]() {
          const auto payload = parse_ir_source_payload(ir_source);

          const auto target = python_path_string(target_path, "ir_to_onnx");
          const auto options = normalize_onnx_binary_write_options(
              target,
              external_data,
              external_data_file,
              external_data_size_threshold);

          return mlx::onnx::ir_to_onnx(
              target,
              payload.dump(),
              normalize_positive_integer(opset, "opset"),
              non_empty_model_name(std::move(model_name)),
              options);
        });
      },
      "target_path"_a,
      "ir_source"_a,
      nb::kw_only(),
      "opset"_a = 18,
      "model_name"_a = std::string("mlx_graph"),
      "external_data"_a = false,
      "external_data_file"_a = nb::none(),
      "external_data_size_threshold"_a = 1024,
      nb::sig(
          "def ir_to_onnx(target_path: Union[str, PathLike], ir_source: Union[dict, list, tuple, str, PathLike], *, opset: int = 18, model_name: str = 'mlx_graph', external_data: bool = False, external_data_file: Optional[str] = None, external_data_size_threshold: int = 1024) -> str"));

  m.def(
      "ir_compatibility_report_json",
      [](nb::handle ir_source) {
        return with_ir_exception_translation<std::string>([&]() {
          const auto payload = parse_ir_source_payload(ir_source);
          return mlx::onnx::ir_compatibility_report_json(payload.dump());
        });
      },
      "ir_source"_a,
      nb::sig(
          "def ir_compatibility_report_json(ir_source: Union[dict, list, tuple, str, PathLike]) -> str"));
}

#include "detail.hpp"
#include "mappings.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>


namespace mlx::onnx::detail {

NameSet collect_payload_tensor_names(const OrderedJson& payload) {
  // Used to reserve namespace for auxiliary tensors introduced during lowering.
  NameSet names;

  for_each_declared_payload_tensor(payload, [&](const OrderedJson& tensor) {
    names.insert(tensor.at("name").get<std::string>());
  });
  for (const auto& node : payload.at("nodes")) {
    for (const auto& name : node.at("inputs")) {
      names.insert(name.get<std::string>());
    }
    for (const auto& name : node.at("outputs")) {
      names.insert(name.get<std::string>());
    }
  }

  return names;
}

} // namespace mlx::onnx::detail


namespace mlx::onnx::detail {

ShapeMap collect_known_tensor_shapes(const OrderedJson& payload) {
  // Seed shape facts from payload metadata; additional facts are inferred while
  // lowering each node.
  ShapeMap out;

  for_each_declared_payload_tensor(payload, [&](const OrderedJson& tensor) {
    out[tensor.at("name").get<std::string>()] =
        normalize_integer_vector(tensor.at("shape"), "tensor shape");
  });

  return out;
}

DtypeMap collect_known_tensor_dtypes(const OrderedJson& payload) {
  // Mirror shape seeding with dtype facts.
  DtypeMap out;

  for_each_declared_payload_tensor(payload, [&](const OrderedJson& tensor) {
    out[tensor.at("name").get<std::string>()] =
        onnx_effective_dtype(tensor.at("dtype").get<std::string>());
  });

  return out;
}

} // namespace mlx::onnx::detail



namespace mlx::onnx {
namespace detail {

static OrderedJson json_from_shape(const Shape& shape) {
  OrderedJson out = OrderedJson::array();
  for (const auto dim : shape) {
    out.push_back(dim);
  }
  return out;
}

static OrderedJson json_from_string_vector(
    const std::vector<std::string>& values) {
  OrderedJson out = OrderedJson::array();
  for (const auto& value : values) {
    out.push_back(value);
  }
  return out;
}

static OrderedJson json_from_int_vector(const std::vector<int64_t>& values) {
  OrderedJson out = OrderedJson::array();
  for (const auto value : values) {
    out.push_back(value);
  }
  return out;
}

static OrderedJson json_from_double_vector(const std::vector<double>& values) {
  OrderedJson out = OrderedJson::array();
  for (const auto value : values) {
    out.push_back(value);
  }
  return out;
}

static std::optional<Shape> known_shape_for(
    const ShapeMap& known_shapes,
    const std::string& name) {
  const auto it = known_shapes.find(name);
  if (it == known_shapes.end()) {
    return std::nullopt;
  }
  return it->second;
}

static std::optional<std::string> known_dtype_for(
    const DtypeMap& known_dtypes,
    const std::string& name) {
  const auto it = known_dtypes.find(name);
  if (it == known_dtypes.end()) {
    return std::nullopt;
  }
  return it->second;
}

static std::string onnx_op_name(const std::string& op) {
  const auto mapped = onnx_op_type_for_ir_op(op);
  if (mapped.has_value()) {
    return mapped.value();
  }

  std::ostringstream out;
  out << "[ir.lowering] unsupported op " << op;
  throw std::runtime_error(out.str());
}

static int64_t
normalize_axis(int64_t axis, size_t rank, const std::string& label) {
  int64_t index = axis;
  if (index < 0) {
    index += static_cast<int64_t>(rank);
  }
  if (index < 0 || index >= static_cast<int64_t>(rank)) {
    std::ostringstream out;
    out << "[ir.lowering] " << label << " " << axis
        << " is out of bounds for rank " << rank;
    throw std::invalid_argument(out.str());
  }
  return index;
}

static std::string unique_aux_tensor_name(
    NameSet& used_tensor_names,
    size_t node_index,
    const std::string& label) {
  // Auxiliary tensors (shape/axes/cast/intermediate nodes) must not collide
  // with user-visible graph names.
  const std::string base =
      "__mlxir_aux_node" + std::to_string(node_index) + "_" + label;
  std::string candidate = base;
  size_t suffix = 0;
  while (used_tensor_names.find(candidate) != used_tensor_names.end()) {
    ++suffix;
    candidate = base + "_" + std::to_string(suffix);
  }
  used_tensor_names.insert(candidate);
  return candidate;
}

static OrderedJson build_onnx_node_spec(
    const std::string& name,
    const std::string& op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    OrderedJson attributes) {
  OrderedJson out = OrderedJson::object();
  out["name"] = name;
  out["op_type"] = op_type;
  out["inputs"] = json_from_string_vector(inputs);
  out["outputs"] = json_from_string_vector(outputs);
  out["attributes"] = std::move(attributes);
  return out;
}

static OrderedJson normalize_initializer_int64_values(
    const OrderedJson& value,
    const std::string& label) {
  if (value.is_array()) {
    OrderedJson out = OrderedJson::array();
    for (const auto& item : value) {
      out.push_back(normalize_initializer_int64_values(item, label));
    }
    return out;
  }
  return OrderedJson(normalized_integer_scalar(value, label));
}

OrderedJson onnx_value_info(const OrderedJson& tensor) {
  // ValueInfo fields are normalized once here so both JSON stub and protobuf
  // encoding use the same inferred elem type symbol.
  OrderedJson out = OrderedJson::object();
  const auto dtype =
      onnx_effective_dtype(tensor.at("dtype").get<std::string>());
  out["name"] = tensor.at("name").get<std::string>();
  out["shape"] = tensor.at("shape");
  out["dtype"] = dtype;
  out["onnx_elem_type"] = onnx_dtype_symbol(dtype);
  return out;
}

OrderedJson onnx_initializer_info(const OrderedJson& tensor) {
  // int64 initializers are normalized recursively so mixed numeric encodings
  // (e.g. unsigned JSON literals) cannot leak into protobuf encoding.
  OrderedJson info = onnx_value_info(tensor);
  OrderedJson values = tensor.at("values");
  if (info.at("dtype").get<std::string>() == "int64") {
    const std::string label =
        "initializer " + info.at("name").get<std::string>();
    values = normalize_initializer_int64_values(values, label);
  }
  info["values"] = std::move(values);
  return info;
}

struct ConvolutionAttributes {
  // Parsed IR convolution semantics (NHWC in IR conventions).
  std::vector<int64_t> strides;
  std::vector<int64_t> padding_low;
  std::vector<int64_t> padding_high;
  std::vector<int64_t> pads;
  std::vector<int64_t> kernel_dilation;
  std::vector<int64_t> input_dilation;
  int64_t groups;
  bool flip;
  size_t spatial_rank;
};

struct ConvTransposeAttributes {
  // ONNX ConvTranspose expects derived padding/output_padding fields.
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  std::vector<int64_t> pads_begin;
  std::vector<int64_t> pads_end;
  std::vector<int64_t> pads;
  std::vector<int64_t> output_padding;
};

struct ScanArguments {
  int64_t reduce_type;
  int64_t axis;
  bool reverse;
  bool inclusive;
};

struct AsStridedArguments {
  Shape output_shape;
  std::vector<int64_t> strides;
  int64_t offset;
};

struct ArangeArguments {
  bool integral;
  int64_t start_i;
  int64_t stop_i;
  int64_t step_i;
  double start_f;
  double stop_f;
  double step_f;
  std::string dtype;
};

struct RandomBitsArguments {
  Shape shape;
  int64_t width;
  std::string dtype;
  double low;
  double high;
};

struct LayerNormArguments {
  double epsilon;
};

struct RoPEArguments {
  int64_t dims;
  bool traditional;
  double base;
  double scale;
  bool forward;
};

struct GatherStateArguments {
  std::vector<int64_t> axes;
  Shape slice_sizes;
};

static std::optional<std::vector<int64_t>> transpose_perm_from_arguments(
    const OrderedJson& arguments) {
  // Transpose argument shape varies across traces; pick first valid int-vector.
  if (!arguments.is_array()) {
    return std::nullopt;
  }

  for (const auto& value : arguments) {
    if (!value.is_array()) {
      continue;
    }

    try {
      return normalize_integer_vector(value, "Transpose permutation");
    } catch (const std::exception&) {
      // Try next argument.
    }
  }

  return std::nullopt;
}

static std::optional<int64_t> concatenate_axis_from_arguments(
    const OrderedJson& arguments,
    bool strict) {
  if (arguments.is_array() && arguments.size() == 1) {
    try {
      return normalized_integer_scalar(arguments.at(0), "Concatenate axis");
    } catch (const std::exception&) {
      // Handled below.
    }
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[ir.lowering] unsupported Concatenate arguments "
      << arguments.dump() << "; expected [axis]";
  throw std::runtime_error(out.str());
}

static std::optional<int64_t> gather_axis_from_arguments(
    const OrderedJson& arguments,
    bool strict) {
  // Gather traces can encode axis either as scalar or single-item vector.
  if (arguments.is_array() && !arguments.empty()) {
    const auto& first = arguments.at(0);
    try {
      return normalized_integer_scalar(first, "Gather axis");
    } catch (const std::exception&) {
      // Try vector-encoded axis.
    }

    if (first.is_array() && first.size() == 1) {
      try {
        return normalized_integer_scalar(first.at(0), "Gather axis");
      } catch (const std::exception&) {
        // Handled below.
      }
    }
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[ir.lowering] unsupported Gather arguments " << arguments.dump()
      << "; expected first argument to encode axis";
  throw std::runtime_error(out.str());
}

static std::optional<GatherStateArguments> gather_state_from_arguments(
    const OrderedJson& arguments,
    bool strict) {
  if (arguments.is_array() && arguments.size() >= 2) {
    try {
      auto axes = normalize_integer_vector(arguments.at(0), "Gather axes");
      auto slice_sizes =
          normalize_integer_vector(arguments.at(1), "Gather slice_sizes");
      return GatherStateArguments{
          std::move(axes), std::move(slice_sizes)};
    } catch (const std::exception&) {
      // Handled below.
    }
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[ir.lowering] unsupported Gather arguments " << arguments.dump()
      << "; expected [axis] or [axes, slice_sizes]";
  throw std::runtime_error(out.str());
}

static std::optional<OrderedJson> scatter_axis_attributes_from_arguments(
    const OrderedJson& arguments,
    bool strict) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    if (!strict) {
      return std::nullopt;
    }
    std::ostringstream out;
    out << "[ir.lowering] unsupported ScatterAxis arguments "
        << arguments.dump() << "; expected [mode, axis]";
    throw std::runtime_error(out.str());
  }

  int64_t mode = 0;
  int64_t axis = 0;
  try {
    mode = normalized_integer_scalar(arguments.at(0), "ScatterAxis mode");
    axis = normalized_integer_scalar(arguments.at(1), "ScatterAxis axis");
  } catch (const std::exception&) {
    if (!strict) {
      return std::nullopt;
    }
    std::ostringstream out;
    out << "[ir.lowering] unsupported ScatterAxis arguments "
        << arguments.dump() << "; mode/axis must be Integer";
    throw std::runtime_error(out.str());
  }

  if (mode != 1) {
    if (!strict) {
      return std::nullopt;
    }

    std::ostringstream out;
    out << "[ir.lowering] unsupported ScatterAxis mode " << mode
        << "; only update mode (1) is supported";
    throw std::runtime_error(out.str());
  }

  OrderedJson attributes = OrderedJson::object();
  attributes["axis"] = axis;
  return attributes;
}

static OrderedJson onnx_node_attributes(const OrderedJson& node) {
  // Attribute canonicalization stays separated from op rewrites so
  // compatibility probing can use the same attribute logic.
  const auto op = node.at("op").get<std::string>();
  const OrderedJson arguments =
      node.contains("arguments") ? node.at("arguments") : OrderedJson::array();

  if (op == "Transpose") {
    const auto perm = transpose_perm_from_arguments(arguments);
    if (!perm.has_value() || perm->empty()) {
      return OrderedJson::object();
    }
    OrderedJson out = OrderedJson::object();
    out["perm"] = json_from_int_vector(*perm);
    return out;
  }

  if (op == "Concatenate") {
    OrderedJson out = OrderedJson::object();
    out["axis"] = concatenate_axis_from_arguments(arguments, true).value();
    return out;
  }

  if (op == "Gather") {
    OrderedJson out = OrderedJson::object();
    if (const auto axis = gather_axis_from_arguments(arguments, false);
        axis.has_value()) {
      out["axis"] = axis.value();
      return out;
    }
    if (!gather_state_from_arguments(arguments, false).has_value()) {
      gather_state_from_arguments(arguments, true);
    }
    return out;
  }

  if (op == "GatherAxis") {
    OrderedJson out = OrderedJson::object();
    out["axis"] = gather_axis_from_arguments(arguments, true).value();
    return out;
  }

  if (op == "ScatterAxis") {
    return scatter_axis_attributes_from_arguments(arguments, true).value();
  }

  return OrderedJson::object();
}

static std::optional<ConvolutionAttributes>
convolution_attributes_from_arguments(
    const OrderedJson& arguments,
    bool strict) {
  // IR convolution argument contract:
  // [strides, padding_low, padding_high, kernel_dilation, input_dilation,
  // groups, flip]
  if (!(arguments.is_array() && arguments.size() >= 7)) {
    if (!strict) {
      return std::nullopt;
    }
    std::ostringstream out;
    out << "[ir.lowering] unsupported Convolution arguments "
        << arguments.dump()
        << "; expected [strides, padding_low, padding_high, kernel_dilation, input_dilation, groups, flip]";
    throw std::runtime_error(out.str());
  }

  ConvolutionAttributes out;
  out.strides =
      normalize_integer_vector(arguments.at(0), "Convolution strides");
  out.padding_low =
      normalize_integer_vector(arguments.at(1), "Convolution padding_low");
  out.padding_high =
      normalize_integer_vector(arguments.at(2), "Convolution padding_high");
  out.kernel_dilation =
      normalize_integer_vector(arguments.at(3), "Convolution kernel_dilation");
  out.input_dilation =
      normalize_integer_vector(arguments.at(4), "Convolution input_dilation");
  out.groups = normalized_integer_scalar(arguments.at(5), "Convolution groups");
  if (!arguments.at(6).is_boolean()) {
    throw std::invalid_argument(
        "[ir.lowering] Convolution flip must be boolean");
  }
  out.flip = arguments.at(6).get<bool>();
  out.spatial_rank = out.strides.size();

  const std::vector<size_t> lengths = {
      out.padding_low.size(),
      out.padding_high.size(),
      out.kernel_dilation.size(),
      out.input_dilation.size()};
  for (size_t length : lengths) {
    if (length != out.spatial_rank) {
      std::ostringstream msg;
      msg << "[ir.lowering] Convolution argument lengths must match spatial rank "
          << out.spatial_rank;
      throw std::invalid_argument(msg.str());
    }
  }

  if (std::any_of(out.strides.begin(), out.strides.end(), [](int64_t value) {
        return value <= 0;
      })) {
    throw std::invalid_argument(
        "[ir.lowering] Convolution strides must be positive");
  }
  if (std::any_of(
          out.padding_low.begin(),
          out.padding_low.end(),
          [](int64_t value) { return value < 0; }) ||
      std::any_of(
          out.padding_high.begin(), out.padding_high.end(), [](int64_t value) {
            return value < 0;
          })) {
    throw std::runtime_error(
        "[ir.lowering] unsupported Convolution with negative padding");
  }
  if (std::any_of(
          out.kernel_dilation.begin(),
          out.kernel_dilation.end(),
          [](int64_t value) { return value <= 0; })) {
    throw std::invalid_argument(
        "[ir.lowering] Convolution kernel_dilation must be positive");
  }
  if (std::any_of(
          out.input_dilation.begin(),
          out.input_dilation.end(),
          [](int64_t value) { return value <= 0; })) {
    throw std::invalid_argument(
        "[ir.lowering] Convolution input_dilation must be positive");
  }
  if (out.groups <= 0) {
    throw std::invalid_argument(
        "[ir.lowering] Convolution groups must be a positive Integer");
  }

  out.pads = out.padding_low;
  out.pads.insert(
      out.pads.end(), out.padding_high.begin(), out.padding_high.end());
  return out;
}

static ConvTransposeAttributes convtranspose_attributes_from_convolution(
    const ConvolutionAttributes& convolution,
    const Shape& weight_shape) {
  // Convert IR "flip=true" convolution semantics into ONNX ConvTranspose
  // parameters. This derives pads/output_padding from low/high paddings.
  const auto weight = normalize_integer_vector(
      json_from_int_vector(weight_shape), "ConvolutionTranspose weight shape");
  const size_t spatial_rank = convolution.spatial_rank;
  const size_t expected_rank = spatial_rank + 2;
  if (weight.size() != expected_rank) {
    std::ostringstream out;
    out << "[ir.lowering] ConvolutionTranspose weight rank mismatch: expected "
        << expected_rank << ", got " << weight.size();
    throw std::invalid_argument(out.str());
  }

  std::vector<int64_t> kernel_shape(weight.begin() + 1, weight.end() - 1);
  const auto& kernel_dilation = convolution.kernel_dilation;
  const auto& padding_low = convolution.padding_low;
  const auto& padding_high = convolution.padding_high;
  const auto& strides = convolution.input_dilation;

  std::vector<int64_t> base_padding;
  base_padding.reserve(spatial_rank);
  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    base_padding.push_back(kernel_dilation[axis] * (kernel_shape[axis] - 1));
  }

  ConvTransposeAttributes out;
  out.strides = strides;
  out.dilations = kernel_dilation;
  out.pads_begin.reserve(spatial_rank);
  out.pads_end.reserve(spatial_rank);
  out.output_padding.reserve(spatial_rank);

  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    out.pads_begin.push_back(base_padding[axis] - padding_low[axis]);
    out.output_padding.push_back(padding_high[axis] - padding_low[axis]);
    out.pads_end.push_back(
        base_padding[axis] - padding_high[axis] + out.output_padding[axis]);
  }

  if (std::any_of(
          out.pads_begin.begin(),
          out.pads_begin.end(),
          [](int64_t value) { return value < 0; }) ||
      std::any_of(out.pads_end.begin(), out.pads_end.end(), [](int64_t value) {
        return value < 0;
      })) {
    throw std::runtime_error(
        "[ir.lowering] unsupported ConvolutionTranspose derived negative padding from arguments");
  }
  if (std::any_of(
          out.output_padding.begin(),
          out.output_padding.end(),
          [](int64_t value) { return value < 0; })) {
    throw std::runtime_error(
        "[ir.lowering] unsupported ConvolutionTranspose with negative output_padding");
  }

  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    if (out.output_padding[axis] >= out.strides[axis]) {
      std::ostringstream msg;
      msg << "[ir.lowering] unsupported ConvolutionTranspose output_padding "
          << json_from_int_vector(out.output_padding).dump()
          << "; each value must be < corresponding stride "
          << json_from_int_vector(out.strides).dump();
      throw std::runtime_error(msg.str());
    }
  }

  out.pads = out.pads_begin;
  out.pads.insert(out.pads.end(), out.pads_end.begin(), out.pads_end.end());
  return out;
}

static std::optional<std::string> reduce_onnx_op_type(
    const OrderedJson& arguments,
    bool strict) {
  int64_t reduce_code = 0;
  if (arguments.is_array() && !arguments.empty()) {
    reduce_code = normalized_integer_scalar(arguments.at(0), "Reduce code");
  } else if (!strict) {
    return std::nullopt;
  }

  const auto mapped = onnx_reduce_op_type_from_code(reduce_code);
  if (mapped.has_value()) {
    return mapped;
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[ir.lowering] unsupported Reduce code " << reduce_code;
  throw std::runtime_error(out.str());
}

static std::optional<std::string> argreduce_onnx_op_type(
    const OrderedJson& arguments,
    bool strict) {
  int64_t reduce_code = 0;
  if (arguments.is_array() && !arguments.empty()) {
    reduce_code = normalized_integer_scalar(arguments.at(0), "ArgReduce code");
  } else if (!strict) {
    return std::nullopt;
  }

  const auto mapped = onnx_argreduce_op_type_from_code(reduce_code);
  if (mapped.has_value()) {
    return mapped;
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[ir.lowering] unsupported ArgReduce code " << reduce_code;
  throw std::runtime_error(out.str());
}

static std::optional<Shape> infer_elementwise_output_shape(
    const std::optional<Shape>& lhs_shape,
    const std::optional<Shape>& rhs_shape) {
  // Numpy-style broadcast shape inference for binary elementwise ops.
  if (!lhs_shape.has_value() || !rhs_shape.has_value()) {
    return std::nullopt;
  }

  const auto& lhs = lhs_shape.value();
  const auto& rhs = rhs_shape.value();
  const size_t max_rank = std::max(lhs.size(), rhs.size());

  Shape lhs_aligned(max_rank - lhs.size(), 1);
  lhs_aligned.insert(lhs_aligned.end(), lhs.begin(), lhs.end());

  Shape rhs_aligned(max_rank - rhs.size(), 1);
  rhs_aligned.insert(rhs_aligned.end(), rhs.begin(), rhs.end());

  Shape out;
  out.reserve(max_rank);

  for (size_t i = 0; i < max_rank; ++i) {
    const auto left = lhs_aligned[i];
    const auto right = rhs_aligned[i];
    if (left == right) {
      out.push_back(left);
    } else if (left == 1) {
      out.push_back(right);
    } else if (right == 1) {
      out.push_back(left);
    } else {
      return std::nullopt;
    }
  }

  return out;
}

static std::optional<Shape> infer_matmul_output_shape(
    const std::optional<Shape>& lhs_shape,
    const std::optional<Shape>& rhs_shape) {
  // ONNX MatMul shape inference with 1D operand normalization rules.
  if (!lhs_shape.has_value() || !rhs_shape.has_value()) {
    return std::nullopt;
  }

  const Shape& lhs = lhs_shape.value();
  const Shape& rhs = rhs_shape.value();
  if (lhs.empty() || rhs.empty()) {
    return std::nullopt;
  }

  const bool lhs_was_1d = lhs.size() == 1;
  const bool rhs_was_1d = rhs.size() == 1;

  Shape lhs_matrix = lhs_was_1d
      ? Shape{1, lhs[0]}
      : Shape{lhs[lhs.size() - 2], lhs[lhs.size() - 1]};
  Shape rhs_matrix = rhs_was_1d
      ? Shape{rhs[0], 1}
      : Shape{rhs[rhs.size() - 2], rhs[rhs.size() - 1]};

  if (lhs_matrix[1] != rhs_matrix[0]) {
    return std::nullopt;
  }

  Shape lhs_batch = lhs_was_1d ? Shape{} : Shape(lhs.begin(), lhs.end() - 2);
  Shape rhs_batch = rhs_was_1d ? Shape{} : Shape(rhs.begin(), rhs.end() - 2);
  const auto batch = infer_elementwise_output_shape(lhs_batch, rhs_batch);
  if (!batch.has_value()) {
    return std::nullopt;
  }

  Shape out = batch.value();
  out.push_back(lhs_matrix[0]);
  out.push_back(rhs_matrix[1]);

  if (lhs_was_1d) {
    out.erase(out.begin() + static_cast<long>(batch->size()));
  }
  if (rhs_was_1d) {
    out.pop_back();
  }

  return out;
}

static std::optional<std::string> promote_binary_dtype(
    const std::optional<std::string>& lhs_dtype,
    const std::optional<std::string>& rhs_dtype) {
  // Local promotion rank table approximates IR expectations for mixed
  // numeric ops where ONNX would otherwise require explicit casts.
  const auto lhs = canonical_dtype(lhs_dtype);
  const auto rhs = canonical_dtype(rhs_dtype);

  if (!lhs.has_value()) {
    return rhs;
  }
  if (!rhs.has_value()) {
    return lhs;
  }
  if (lhs.value() == rhs.value()) {
    return lhs;
  }

  const auto lhs_rank = dtype_promotion_rank(lhs.value());
  const auto rhs_rank = dtype_promotion_rank(rhs.value());
  if (!lhs_rank.has_value() || !rhs_rank.has_value()) {
    return lhs;
  }

  return lhs_rank.value() >= rhs_rank.value() ? lhs : rhs;
}

static int64_t normalize_slice_index(int64_t value, int64_t dim) {
  int64_t index = value;
  if (index < 0) {
    index += dim;
  }
  if (index < 0) {
    index = 0;
  }
  if (index > dim) {
    index = dim;
  }
  return index;
}

static std::vector<int64_t> reduce_axes_from_arguments(
    const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument(
        "[ir.lowering] Reduce arguments must include reduction code and axes");
  }
  return normalize_integer_vector(arguments.at(1), "Reduce axes");
}

static std::optional<Shape> infer_reduce_keepdims_shape(
    const std::optional<Shape>& input_shape,
    const std::vector<int64_t>& axes) {
  // Lowering keeps keepdims=1 for currently supported reductions.
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  Shape shape = input_shape.value();
  const size_t rank = shape.size();

  std::set<int64_t> normalized_axes;
  for (const auto axis : axes) {
    normalized_axes.insert(normalize_axis(axis, rank, "Reduce axis"));
  }

  for (const auto axis : normalized_axes) {
    shape[static_cast<size_t>(axis)] = 1;
  }

  return shape;
}

static std::string as_type_target_dtype(
    const OrderedJson& arguments,
    const std::vector<std::string>& outputs,
    const DtypeMap& known_dtypes) {
  // Prefer explicit dtype argument; fallback to already-known output dtype only
  // when unambiguous.
  if (arguments.is_array() && !arguments.empty()) {
    const auto& target = arguments.at(0);
    bool valid_dtype = false;
    if (target.is_string()) {
      try {
        const auto normalized = onnx_effective_dtype(target.get<std::string>());
        (void)onnx_dtype_symbol(normalized);
        valid_dtype = true;
      } catch (const std::exception&) {
        valid_dtype = false;
      }
    }
    if (!valid_dtype) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported AsType arguments "
          << arguments.dump() << "; expected first argument to be dtype String";
      throw std::runtime_error(out.str());
    }
    return target.get<std::string>();
  }

  std::set<std::string> candidates;
  for (const auto& name : outputs) {
    const auto it = known_dtypes.find(name);
    if (it != known_dtypes.end()) {
      candidates.insert(it->second);
    }
  }

  if (candidates.size() == 1) {
    return *candidates.begin();
  }
  if (candidates.size() > 1) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported AsType with inconsistent output dtypes";
    throw std::runtime_error(out.str());
  }

  throw std::runtime_error(
      "[ir.lowering] unsupported AsType without target dtype argument");
}

static bool equal_nan_from_arguments(const OrderedJson& arguments) {
  if (!arguments.is_array() || arguments.empty()) {
    return false;
  }
  if (!(arguments.size() == 1 && arguments.at(0).is_boolean())) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported Equal arguments "
        << arguments.dump() << "; expected [equal_nan]";
    throw std::runtime_error(out.str());
  }
  return arguments.at(0).get<bool>();
}

static LayerNormArguments layernorm_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 1 &&
        json_is_numeric(arguments.at(0)))) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported LayerNorm arguments "
        << arguments.dump() << "; expected [epsilon]";
    throw std::runtime_error(out.str());
  }

  const auto& epsilon_value = arguments.at(0);
  double epsilon = 0.0;
  if (epsilon_value.is_number_float()) {
    epsilon = epsilon_value.get<double>();
  } else {
    epsilon = static_cast<double>(
        normalized_integer_scalar(epsilon_value, "LayerNorm epsilon"));
  }

  return {epsilon};
}

static RoPEArguments rope_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 5)) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported RoPE arguments " << arguments.dump()
        << "; expected [dims, traditional, base, scale, forward]";
    throw std::runtime_error(out.str());
  }

  const auto dims = normalized_integer_scalar(arguments.at(0), "RoPE dims");
  if (dims <= 0 || (dims % 2) != 0) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported RoPE dims " << dims
        << "; expected positive even Integer";
    throw std::runtime_error(out.str());
  }

  if (!arguments.at(1).is_boolean()) {
    throw std::invalid_argument(
        "[ir.lowering] RoPE traditional flag must be boolean");
  }
  if (!json_is_numeric(arguments.at(2))) {
    throw std::invalid_argument(
        "[ir.lowering] RoPE base must be Numeric");
  }
  if (!json_is_numeric(arguments.at(3))) {
    throw std::invalid_argument(
        "[ir.lowering] RoPE scale must be Numeric");
  }
  if (!arguments.at(4).is_boolean()) {
    throw std::invalid_argument(
        "[ir.lowering] RoPE forward flag must be boolean");
  }

  const auto parse_numeric = [](const OrderedJson& value,
                                const std::string& label) {
    if (value.is_number_float()) {
      return value.get<double>();
    }
    return static_cast<double>(normalized_integer_scalar(value, label));
  };

  return {
      dims,
      arguments.at(1).get<bool>(),
      parse_numeric(arguments.at(2), "RoPE base"),
      parse_numeric(arguments.at(3), "RoPE scale"),
      arguments.at(4).get<bool>()};
}

static std::vector<int64_t> infer_logsumexp_axes(
    const Shape& input_shape,
    const std::optional<Shape>& output_shape) {
  // IR trace omits explicit axes for LogSumExp. Infer from output shape,
  // defaulting to last axis semantics.
  const Shape input = input_shape;
  if (output_shape.has_value()) {
    const Shape output = output_shape.value();

    if (output.size() == input.size()) {
      std::vector<int64_t> axes;
      for (size_t i = 0; i < input.size(); ++i) {
        const auto dim = input[i];
        const auto out_dim = output[i];
        if (out_dim == 1 && dim != 1) {
          axes.push_back(static_cast<int64_t>(i));
        } else if (out_dim != dim) {
          std::ostringstream out;
          out << "[ir.lowering] unsupported LogSumExp output shape "
              << json_from_shape(output).dump() << " for input "
              << json_from_shape(input).dump();
          throw std::runtime_error(out.str());
        }
      }
      if (axes.empty()) {
        return {static_cast<int64_t>(input.size() - 1)};
      }
      return axes;
    }

    if (output.size() == input.size() - 1) {
      return {static_cast<int64_t>(input.size() - 1)};
    }
  }

  return {static_cast<int64_t>(input.size() - 1)};
}

static std::
    tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
    pad_axes_and_sizes_from_arguments(
        const OrderedJson& arguments,
        const Shape& input_shape) {
  // IR Pad uses sparse axis specification; ONNX expects full-rank pads.
  if (!(arguments.is_array() && arguments.size() >= 3)) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported Pad arguments " << arguments.dump()
        << "; expected [axes, low, high]";
    throw std::runtime_error(out.str());
  }

  auto axes = normalize_integer_vector(arguments.at(0), "Pad axes");
  auto low = normalize_integer_vector(arguments.at(1), "Pad low");
  auto high = normalize_integer_vector(arguments.at(2), "Pad high");

  if (!(axes.size() == low.size() && low.size() == high.size())) {
    std::ostringstream out;
    out << "[ir.lowering] Pad axes/low/high lengths must match: "
        << axes.size() << "/" << low.size() << "/" << high.size();
    throw std::invalid_argument(out.str());
  }

  if (std::any_of(
          low.begin(), low.end(), [](int64_t value) { return value < 0; }) ||
      std::any_of(
          high.begin(), high.end(), [](int64_t value) { return value < 0; })) {
    throw std::runtime_error(
        "[ir.lowering] unsupported Pad with negative padding");
  }

  const size_t rank = input_shape.size();
  std::vector<int64_t> normalized_axes;
  normalized_axes.reserve(axes.size());
  for (const auto axis : axes) {
    normalized_axes.push_back(normalize_axis(axis, rank, "Pad axis"));
  }

  std::set<int64_t> uniq(normalized_axes.begin(), normalized_axes.end());
  if (uniq.size() != normalized_axes.size()) {
    throw std::invalid_argument(
        "[ir.lowering] Pad axes must not contain duplicates");
  }

  return {normalized_axes, low, high};
}

static std::optional<Shape> infer_pad_output_shape(
    const std::optional<Shape>& input_shape,
    const std::vector<int64_t>& pads_begin,
    const std::vector<int64_t>& pads_end) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  const Shape shape = input_shape.value();
  if (!(pads_begin.size() == shape.size() && pads_end.size() == shape.size())) {
    std::ostringstream out;
    out << "[ir.lowering] Pad low/high ranks must match input rank "
        << shape.size();
    throw std::invalid_argument(out.str());
  }

  Shape out;
  out.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    out.push_back(shape[i] + pads_begin[i] + pads_end[i]);
  }
  return out;
}

static ScanArguments scan_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 4)) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported Scan arguments " << arguments.dump()
        << "; expected [reduce_type, axis, reverse, inclusive]";
    throw std::runtime_error(out.str());
  }

  const auto reduce_type =
      normalized_integer_scalar(arguments.at(0), "Scan reduce_type");
  const auto axis = normalized_integer_scalar(arguments.at(1), "Scan axis");
  if (!arguments.at(2).is_boolean()) {
    throw std::invalid_argument(
        "[ir.lowering] Scan reverse must be boolean");
  }
  if (!arguments.at(3).is_boolean()) {
    throw std::invalid_argument(
        "[ir.lowering] Scan inclusive must be boolean");
  }

  return {
      reduce_type,
      axis,
      arguments.at(2).get<bool>(),
      arguments.at(3).get<bool>()};
}

static std::pair<int64_t, int64_t> argreduce_mode_axis(
    const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument(
        "[ir.lowering] ArgReduce arguments must include [mode, axis]");
  }

  const int64_t mode =
      normalized_integer_scalar(arguments.at(0), "ArgReduce mode");
  const int64_t axis =
      normalized_integer_scalar(arguments.at(1), "ArgReduce axis");
  if (!onnx_argreduce_op_type_from_code(mode).has_value()) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported ArgReduce code " << mode;
    throw std::runtime_error(out.str());
  }

  return {mode, axis};
}

static std::optional<Shape> infer_argreduce_keepdims_shape(
    const std::optional<Shape>& input_shape,
    int64_t axis) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  Shape shape = input_shape.value();
  const auto axis_index = normalize_axis(axis, shape.size(), "ArgReduce axis");
  shape[static_cast<size_t>(axis_index)] = 1;
  return shape;
}

static std::optional<Shape> infer_convolution_output_shape(
    const std::optional<Shape>& input_shape,
    const std::optional<Shape>& weight_shape,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& padding_low,
    const std::vector<int64_t>& padding_high,
    const std::vector<int64_t>& kernel_dilation,
    int64_t groups) {
  if (!input_shape.has_value() || !weight_shape.has_value()) {
    return std::nullopt;
  }

  const Shape input = input_shape.value();
  const Shape weight = weight_shape.value();
  const size_t spatial_rank = strides.size();
  const size_t expected_rank = spatial_rank + 2;

  if (!(input.size() == expected_rank && weight.size() == expected_rank)) {
    return std::nullopt;
  }
  if (!(padding_low.size() == spatial_rank &&
        padding_high.size() == spatial_rank &&
        kernel_dilation.size() == spatial_rank)) {
    return std::nullopt;
  }

  const int64_t batch = input[0];
  const int64_t input_channels = input.back();
  const int64_t output_channels = weight[0];
  const int64_t weight_input_channels = weight.back();
  if (input_channels != weight_input_channels * groups) {
    return std::nullopt;
  }

  Shape output_spatial;
  output_spatial.reserve(spatial_rank);
  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    const int64_t input_dim = input[axis + 1];
    const int64_t kernel_dim = weight[axis + 1];
    const int64_t dilation = kernel_dilation[axis];
    const int64_t stride = strides[axis];
    const int64_t low = padding_low[axis];
    const int64_t high = padding_high[axis];

    if (kernel_dim <= 0) {
      return std::nullopt;
    }

    const int64_t effective_kernel = dilation * (kernel_dim - 1) + 1;
    const int64_t numerator = input_dim + low + high - effective_kernel;
    if (numerator < 0) {
      return std::nullopt;
    }

    output_spatial.push_back((numerator / stride) + 1);
  }

  Shape out;
  out.push_back(batch);
  out.insert(out.end(), output_spatial.begin(), output_spatial.end());
  out.push_back(output_channels);
  return out;
}

static std::optional<Shape> infer_convolution_transpose_output_shape(
    const std::optional<Shape>& input_shape,
    const std::optional<Shape>& weight_shape,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads_begin,
    const std::vector<int64_t>& pads_end,
    const std::vector<int64_t>& kernel_dilation,
    const std::vector<int64_t>& output_padding,
    int64_t groups) {
  if (!input_shape.has_value() || !weight_shape.has_value()) {
    return std::nullopt;
  }

  const Shape input = input_shape.value();
  const Shape weight = weight_shape.value();
  const size_t spatial_rank = strides.size();
  const size_t expected_rank = spatial_rank + 2;

  if (!(input.size() == expected_rank && weight.size() == expected_rank)) {
    return std::nullopt;
  }
  if (!(pads_begin.size() == spatial_rank && pads_end.size() == spatial_rank &&
        kernel_dilation.size() == spatial_rank &&
        output_padding.size() == spatial_rank)) {
    return std::nullopt;
  }

  const int64_t batch = input[0];
  const int64_t input_channels = input.back();
  const int64_t output_channels = weight[0];
  const int64_t weight_input_channels = weight.back();
  if (input_channels != weight_input_channels * groups) {
    return std::nullopt;
  }

  Shape output_spatial;
  output_spatial.reserve(spatial_rank);
  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    const int64_t input_dim = input[axis + 1];
    const int64_t kernel_dim = weight[axis + 1];
    const int64_t dilation = kernel_dilation[axis];
    const int64_t stride = strides[axis];
    const int64_t low = pads_begin[axis];
    const int64_t high = pads_end[axis];
    const int64_t out_padding = output_padding[axis];

    if (kernel_dim <= 0) {
      return std::nullopt;
    }

    const int64_t effective_kernel = dilation * (kernel_dim - 1) + 1;
    const int64_t dim =
        stride * (input_dim - 1) + out_padding + effective_kernel - low - high;
    if (dim < 0) {
      return std::nullopt;
    }

    output_spatial.push_back(dim);
  }

  Shape out;
  out.push_back(batch);
  out.insert(out.end(), output_spatial.begin(), output_spatial.end());
  out.push_back(output_channels);
  return out;
}

static std::optional<Shape> infer_gather_output_shape(
    const std::optional<Shape>& data_shape,
    const std::optional<Shape>& indices_shape,
    int64_t axis) {
  if (!data_shape.has_value() || !indices_shape.has_value()) {
    return std::nullopt;
  }

  Shape out;
  const auto& data = data_shape.value();
  const auto& indices = indices_shape.value();

  out.insert(out.end(), data.begin(), data.begin() + axis);
  out.insert(out.end(), indices.begin(), indices.end());
  out.insert(out.end(), data.begin() + axis + 1, data.end());
  return out;
}

static std::tuple<
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>>
slice_vectors_from_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument(
        "[ir.lowering] Slice arguments must include starts and ends");
  }

  auto starts = normalize_integer_vector(arguments.at(0), "Slice starts");
  auto ends = normalize_integer_vector(arguments.at(1), "Slice ends");
  std::vector<int64_t> steps;
  if (arguments.size() >= 3) {
    steps = normalize_integer_vector(arguments.at(2), "Slice steps");
  } else {
    steps = std::vector<int64_t>(starts.size(), 1);
  }

  if (!(starts.size() == ends.size() && starts.size() == steps.size())) {
    std::ostringstream out;
    out << "[ir.lowering] Slice starts/ends/steps lengths must match: "
        << starts.size() << "/" << ends.size() << "/" << steps.size();
    throw std::invalid_argument(out.str());
  }
  if (std::any_of(steps.begin(), steps.end(), [](int64_t value) {
        return value == 0;
      })) {
    throw std::invalid_argument(
        "[ir.lowering] Slice steps must not contain zero");
  }

  std::vector<int64_t> axes;
  axes.reserve(starts.size());
  for (size_t i = 0; i < starts.size(); ++i) {
    axes.push_back(static_cast<int64_t>(i));
  }

  return {starts, ends, axes, steps};
}

static std::optional<Shape> infer_slice_output_shape(
    const std::optional<Shape>& input_shape,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& axes,
    const std::vector<int64_t>& steps) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  Shape out_shape = input_shape.value();

  for (size_t i = 0; i < axes.size(); ++i) {
    const auto axis_index =
        normalize_axis(axes[i], out_shape.size(), "Slice axis");
    const auto dim = out_shape[static_cast<size_t>(axis_index)];
    const auto start_v = normalize_slice_index(starts[i], dim);
    const auto end_v = normalize_slice_index(ends[i], dim);
    const auto step_v = steps[i];

    if (step_v <= 0) {
      return std::nullopt;
    }

    out_shape[static_cast<size_t>(axis_index)] =
        end_v <= start_v ? 0 : ((end_v - start_v - 1) / step_v) + 1;
  }

  return out_shape;
}

static std::vector<int64_t> split_lengths_from_indices(
    const std::vector<int64_t>& indices,
    int64_t dim) {
  int64_t prev = 0;
  std::vector<int64_t> lengths;
  lengths.reserve(indices.size() + 1);

  for (const auto index : indices) {
    int64_t value = index;
    if (value < 0) {
      value += dim;
    }
    if (value < prev || value > dim) {
      std::ostringstream out;
      out << "[ir.lowering] Split boundary " << index
          << " is out of range or not non-decreasing for dim " << dim;
      throw std::invalid_argument(out.str());
    }
    lengths.push_back(value - prev);
    prev = value;
  }

  lengths.push_back(dim - prev);
  return lengths;
}

static std::pair<int64_t, std::vector<int64_t>> split_axis_and_lengths(
    const OrderedJson& arguments,
    const std::optional<Shape>& input_shape,
    int64_t output_count) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument(
        "[ir.lowering] Split arguments must include split spec and axis");
  }
  if (output_count <= 0) {
    throw std::invalid_argument(
        "[ir.lowering] Split must have at least one output");
  }

  const auto spec = normalize_integer_vector(arguments.at(0), "Split spec");
  const auto axis = normalized_integer_scalar(arguments.at(1), "Split axis");

  if (!input_shape.has_value()) {
    throw std::runtime_error(
        "[ir.lowering] unsupported Split without known input shape");
  }

  const Shape data_shape = input_shape.value();
  const auto axis_index = normalize_axis(axis, data_shape.size(), "Split axis");
  const int64_t dim = data_shape[static_cast<size_t>(axis_index)];

  std::vector<int64_t> lengths;
  if (spec.size() == 1 && spec[0] == output_count) {
    const auto parts = spec[0];
    if (parts <= 0) {
      throw std::invalid_argument(
          "[ir.lowering] Split parts must be positive");
    }

    const auto quotient = dim / parts;
    const auto remainder = dim % parts;
    if (remainder != 0) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported uneven equal Split: dim " << dim
          << " not divisible by " << parts;
      throw std::runtime_error(out.str());
    }
    lengths.assign(static_cast<size_t>(parts), quotient);
  } else if (spec.size() == static_cast<size_t>(output_count - 1)) {
    lengths = split_lengths_from_indices(spec, dim);
  } else if (spec.size() == static_cast<size_t>(output_count)) {
    lengths = spec;
  } else {
    std::ostringstream out;
    out << "[ir.lowering] unsupported Split spec "
        << json_from_int_vector(spec).dump() << " for " << output_count
        << " outputs";
    throw std::runtime_error(out.str());
  }

  if (static_cast<int64_t>(lengths.size()) != output_count) {
    std::ostringstream out;
    out << "[ir.lowering] Split lengths count " << lengths.size()
        << " does not match outputs " << output_count;
    throw std::invalid_argument(out.str());
  }

  if (std::any_of(lengths.begin(), lengths.end(), [](int64_t value) {
        return value < 0;
      })) {
    throw std::invalid_argument(
        "[ir.lowering] Split lengths must be non-negative");
  }

  const int64_t sum =
      std::accumulate(lengths.begin(), lengths.end(), static_cast<int64_t>(0));
  if (sum != dim) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported Split lengths "
        << json_from_int_vector(lengths).dump() << "; expected sum " << dim;
    throw std::runtime_error(out.str());
  }

  return {axis_index, lengths};
}

static std::optional<std::vector<Shape>> infer_split_output_shapes(
    const std::optional<Shape>& input_shape,
    int64_t axis,
    const std::vector<int64_t>& lengths) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  const auto shape = input_shape.value();
  std::vector<Shape> out;
  out.reserve(lengths.size());

  for (const auto length : lengths) {
    Shape current = shape;
    current[static_cast<size_t>(axis)] = length;
    out.push_back(std::move(current));
  }

  return out;
}

static std::optional<Shape> infer_concatenate_output_shape(
    const std::vector<std::optional<Shape>>& input_shapes,
    int64_t axis) {
  if (input_shapes.empty()) {
    return std::nullopt;
  }
  if (std::any_of(
          input_shapes.begin(), input_shapes.end(), [](const auto& item) {
            return !item.has_value();
          })) {
    return std::nullopt;
  }

  const Shape first = input_shapes.front().value();
  const size_t rank = first.size();
  for (const auto& shape : input_shapes) {
    if (shape->size() != rank) {
      return std::nullopt;
    }
  }

  const auto axis_index = normalize_axis(axis, rank, "Concatenate axis");
  Shape out = first;
  out[static_cast<size_t>(axis_index)] = 0;

  for (const auto& shape : input_shapes) {
    for (size_t i = 0; i < rank; ++i) {
      if (static_cast<int64_t>(i) == axis_index) {
        continue;
      }
      if ((*shape)[i] != out[i]) {
        return std::nullopt;
      }
    }
    out[static_cast<size_t>(axis_index)] +=
        (*shape)[static_cast<size_t>(axis_index)];
  }

  return out;
}

static std::optional<Shape> infer_squeeze_output_shape(
    const std::optional<Shape>& input_shape,
    const std::vector<int64_t>& axes) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  Shape shape = input_shape.value();
  const size_t rank = shape.size();

  std::set<int64_t> normalized_axes;
  for (const auto axis : axes) {
    normalized_axes.insert(normalize_axis(axis, rank, "Squeeze axis"));
  }

  for (auto it = normalized_axes.rbegin(); it != normalized_axes.rend(); ++it) {
    const auto axis_index = static_cast<size_t>(*it);
    const auto dim = shape[axis_index];
    if (dim != 1) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported Squeeze axis " << axis_index
          << " for dim " << dim << "; expected dimension 1";
      throw std::runtime_error(out.str());
    }
    shape.erase(shape.begin() + static_cast<long>(axis_index));
  }

  return shape;
}

static std::optional<Shape> infer_unsqueeze_output_shape(
    const std::optional<Shape>& input_shape,
    const std::vector<int64_t>& axes) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  Shape out = input_shape.value();
  const size_t output_rank = out.size() + axes.size();

  std::vector<int64_t> normalized_axes;
  normalized_axes.reserve(axes.size());
  for (const auto axis : axes) {
    int64_t value = axis;
    if (value < 0) {
      value += static_cast<int64_t>(output_rank);
    }
    if (value < 0 || value >= static_cast<int64_t>(output_rank)) {
      std::ostringstream msg;
      msg << "[ir.lowering] ExpandDims axis " << axis
          << " is out of bounds for output rank " << output_rank;
      throw std::invalid_argument(msg.str());
    }
    normalized_axes.push_back(value);
  }

  std::set<int64_t> uniq(normalized_axes.begin(), normalized_axes.end());
  if (uniq.size() != normalized_axes.size()) {
    throw std::invalid_argument(
        "[ir.lowering] ExpandDims axes must not contain duplicates");
  }

  std::sort(normalized_axes.begin(), normalized_axes.end());
  for (const auto axis : normalized_axes) {
    out.insert(out.begin() + static_cast<long>(axis), 1);
  }

  return out;
}

static std::vector<int64_t> gather_reorder_permutation(
    int64_t data_rank,
    int64_t indices_rank,
    int64_t axis) {
  // IR Gather inserts index dimensions at axis; this permutation restores
  // expected ordering before the explicit Unsqueeze step used by this lowering.
  std::vector<int64_t> indices;
  for (int64_t i = axis; i < axis + indices_rank; ++i) {
    indices.push_back(i);
  }

  std::vector<int64_t> prefix;
  for (int64_t i = 0; i < axis; ++i) {
    prefix.push_back(i);
  }

  std::vector<int64_t> suffix;
  const int64_t suffix_start = axis + indices_rank;
  const int64_t suffix_end = data_rank + indices_rank - 2;
  if (suffix_start <= suffix_end) {
    for (int64_t i = suffix_start; i <= suffix_end; ++i) {
      suffix.push_back(i);
    }
  }

  std::vector<int64_t> out;
  out.reserve(indices.size() + prefix.size() + suffix.size());
  out.insert(out.end(), indices.begin(), indices.end());
  out.insert(out.end(), prefix.begin(), prefix.end());
  out.insert(out.end(), suffix.begin(), suffix.end());
  return out;
}

static bool identity_permutation(const std::vector<int64_t>& perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

static std::pair<std::vector<int64_t>, std::vector<int64_t>>
convolution_data_permutations(size_t spatial_rank) {
  // IR uses NHWC-like layout while ONNX Conv/ConvTranspose use NCHW.
  // Return forward and inverse permutations.
  const auto rank = static_cast<int64_t>(spatial_rank + 2);

  std::vector<int64_t> to_onnx = {0, rank - 1};
  for (int64_t i = 1; i < rank - 1; ++i) {
    to_onnx.push_back(i);
  }

  std::vector<int64_t> from_onnx = {0};
  for (int64_t i = 2; i < rank; ++i) {
    from_onnx.push_back(i);
  }
  from_onnx.push_back(1);

  return {to_onnx, from_onnx};
}

static std::vector<int64_t> convolution_weight_permutation(
    size_t spatial_rank) {
  const auto rank = static_cast<int64_t>(spatial_rank + 2);
  std::vector<int64_t> out = {0, rank - 1};
  for (int64_t i = 1; i < rank - 1; ++i) {
    out.push_back(i);
  }
  return out;
}

static std::vector<int64_t> convolution_transpose_weight_permutation(
    size_t spatial_rank) {
  const auto rank = static_cast<int64_t>(spatial_rank + 2);
  std::vector<int64_t> out = {rank - 1, 0};
  for (int64_t i = 1; i < rank - 1; ++i) {
    out.push_back(i);
  }
  return out;
}

static Shape permute_shape(
    const Shape& shape,
    const std::vector<int64_t>& perm,
    const std::string& label) {
  if (perm.size() != shape.size()) {
    std::ostringstream out;
    out << "[ir.lowering] invalid permutation "
        << json_from_int_vector(perm).dump() << " for " << label << " rank "
        << shape.size();
    throw std::invalid_argument(out.str());
  }

  std::vector<int64_t> sorted = perm;
  std::sort(sorted.begin(), sorted.end());
  for (size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i] != static_cast<int64_t>(i)) {
      std::ostringstream out;
      out << "[ir.lowering] invalid permutation "
          << json_from_int_vector(perm).dump() << " for " << label << " rank "
          << shape.size();
      throw std::invalid_argument(out.str());
    }
  }

  Shape out;
  out.reserve(perm.size());
  for (const auto axis : perm) {
    out.push_back(shape[static_cast<size_t>(axis)]);
  }
  return out;
}

static std::vector<int64_t> integer_vector_argument(
    const OrderedJson& arguments,
    const std::string& op_name) {
  if (arguments.is_array()) {
    for (const auto& value : arguments) {
      if (!value.is_array()) {
        continue;
      }

      try {
        return normalize_integer_vector(value, op_name + " argument");
      } catch (const std::exception&) {
        // Try next.
      }
    }
  }

  std::ostringstream out;
  out << "[ir.lowering] " << op_name
      << " requires an integer-vector argument";
  throw std::invalid_argument(out.str());
}

static Shape flatten_shape_from_arguments(
    const OrderedJson& arguments,
    const Shape& input_shape) {
  // IR Flatten semantics are implemented as Reshape with explicit target.
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported Flatten arguments "
        << arguments.dump() << "; expected [start_axis, end_axis]";
    throw std::runtime_error(out.str());
  }

  const auto rank = input_shape.size();
  if (rank <= 0) {
    throw std::invalid_argument(
        "[ir.lowering] Flatten input shape must have rank >= 1");
  }

  const auto start_axis =
      normalized_integer_scalar(arguments.at(0), "Flatten start_axis");
  const auto end_axis =
      normalized_integer_scalar(arguments.at(1), "Flatten end_axis");
  const auto start_index =
      normalize_axis(start_axis, rank, "Flatten start_axis");
  const auto end_index = normalize_axis(end_axis, rank, "Flatten end_axis");
  if (end_index < start_index) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported Flatten axis range "
        << arguments.dump() << " for rank " << rank;
    throw std::runtime_error(out.str());
  }

  Shape out;
  out.insert(out.end(), input_shape.begin(), input_shape.begin() + start_index);

  int64_t middle = 1;
  for (int64_t axis = start_index; axis <= end_index; ++axis) {
    middle *= input_shape[static_cast<size_t>(axis)];
  }
  out.push_back(middle);

  out.insert(out.end(), input_shape.begin() + end_index + 1, input_shape.end());
  return out;
}

static std::pair<int64_t, Shape> unflatten_axis_shape_from_arguments(
    const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument(
        "[ir.lowering] Unflatten arguments must include [axis, shape]");
  }

  const auto axis =
      normalized_integer_scalar(arguments.at(0), "Unflatten axis");
  auto target_shape =
      normalize_integer_vector(arguments.at(1), "Unflatten shape");

  size_t negative_count = 0;
  for (const auto dim : target_shape) {
    if (dim == -1) {
      ++negative_count;
      continue;
    }
    if (dim <= 0) {
      throw std::invalid_argument(
          "[ir.lowering] Unflatten shape values must be positive or -1");
    }
  }
  if (negative_count > 1) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported Unflatten shape "
        << json_from_int_vector(target_shape).dump()
        << "; at most one -1 is allowed";
    throw std::runtime_error(out.str());
  }

  return {axis, std::move(target_shape)};
}

static Shape unflatten_shape_from_arguments(
    const OrderedJson& arguments,
    const Shape& input_shape) {
  // Resolve one optional -1 dimension and splice shape into the chosen axis.
  auto [axis, target_shape] = unflatten_axis_shape_from_arguments(arguments);
  const auto source_rank = input_shape.size();
  const auto axis_index = normalize_axis(axis, source_rank, "Unflatten axis");
  const auto source_dim = input_shape[static_cast<size_t>(axis_index)];

  std::vector<size_t> negative_indices;
  int64_t known_product = 1;

  for (size_t i = 0; i < target_shape.size(); ++i) {
    const auto dim = target_shape[i];
    if (dim == -1) {
      negative_indices.push_back(i);
      continue;
    }
    known_product *= dim;
  }

  if (negative_indices.size() == 1) {
    const auto unknown_index = negative_indices.front();
    if (known_product <= 0 || source_dim % known_product != 0) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported Unflatten shape "
          << json_from_int_vector(target_shape).dump() << " for source dim "
          << source_dim;
      throw std::runtime_error(out.str());
    }
    target_shape[unknown_index] = source_dim / known_product;
  } else if (known_product != source_dim) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported Unflatten shape "
        << json_from_int_vector(target_shape).dump() << "; product "
        << known_product << " must match source dim " << source_dim;
    throw std::runtime_error(out.str());
  }

  Shape out;
  out.insert(out.end(), input_shape.begin(), input_shape.begin() + axis_index);
  out.insert(out.end(), target_shape.begin(), target_shape.end());
  out.insert(
      out.end(), input_shape.begin() + axis_index + 1, input_shape.end());
  return out;
}

static bool integer_like_numeric(const OrderedJson& value) {
  if (value.is_number_integer() || value.is_number_unsigned()) {
    return true;
  }
  if (!value.is_number_float()) {
    return false;
  }
  const double raw = value.get<double>();
  return std::isfinite(raw) && std::trunc(raw) == raw;
}

static RandomBitsArguments random_bits_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument(
        "[ir.lowering] RandomBits arguments must include [shape, width]");
  }

  RandomBitsArguments out;
  out.shape = normalize_integer_vector(arguments.at(0), "RandomBits shape");
  if (std::any_of(out.shape.begin(), out.shape.end(), [](int64_t value) {
        return value < 0;
      })) {
    throw std::invalid_argument(
        "[ir.lowering] RandomBits shape values must be non-negative");
  }

  out.width = normalized_integer_scalar(arguments.at(1), "RandomBits width");
  out.low = 0.0;

  switch (out.width) {
    case 1:
      out.dtype = "uint8";
      break;
    case 2:
      out.dtype = "uint16";
      break;
    case 4:
      out.dtype = "uint32";
      break;
    default: {
      std::ostringstream msg;
      msg << "[ir.lowering] unsupported RandomBits width " << out.width
          << "; expected one of {1, 2, 4}";
      throw std::runtime_error(msg.str());
    }
  }

  out.high = std::ldexp(1.0, static_cast<int>(out.width * 8));
  return out;
}

static ArangeArguments arange_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 3)) {
    throw std::invalid_argument(
        "[ir.lowering] Arange arguments must include [start, stop, step]");
  }

  const auto& start = arguments.at(0);
  const auto& stop = arguments.at(1);
  const auto& step = arguments.at(2);
  if (!(json_is_numeric(start) && json_is_numeric(stop) &&
        json_is_numeric(step))) {
    throw std::invalid_argument(
        "[ir.lowering] Arange start/stop/step must be Numeric");
  }

  ArangeArguments out;
  if (integer_like_numeric(start) && integer_like_numeric(stop) &&
      integer_like_numeric(step)) {
    out.integral = true;
    out.start_i = normalized_integer_scalar(start, "Arange start");
    out.stop_i = normalized_integer_scalar(stop, "Arange stop");
    out.step_i = normalized_integer_scalar(step, "Arange step");
    if (out.step_i == 0) {
      throw std::invalid_argument(
          "[ir.lowering] Arange step must not be zero");
    }
    out.dtype = "int64";
    return out;
  }

  out.integral = false;
  out.start_f = start.get<double>();
  out.stop_f = stop.get<double>();
  out.step_f = step.get<double>();
  if (!(std::isfinite(out.start_f) && std::isfinite(out.stop_f) &&
        std::isfinite(out.step_f))) {
    throw std::invalid_argument(
        "[ir.lowering] Arange start/stop/step must be finite Numeric values");
  }
  if (out.step_f == 0.0) {
    throw std::invalid_argument(
        "[ir.lowering] Arange step must not be zero");
  }
  out.dtype = "float32";
  return out;
}

static OrderedJson arange_values(const ArangeArguments& args) {
  // Materialize arange as an initializer, avoiding backend-dependent Range ops.
  OrderedJson values = OrderedJson::array();
  if (args.integral) {
    int64_t current = args.start_i;
    if (args.step_i > 0) {
      while (current < args.stop_i) {
        values.push_back(current);
        current += args.step_i;
      }
    } else {
      while (current > args.stop_i) {
        values.push_back(current);
        current += args.step_i;
      }
    }
    return values;
  }

  double current = args.start_f;
  if (args.step_f > 0.0) {
    while (current < args.stop_f) {
      values.push_back(current);
      current += args.step_f;
    }
  } else {
    while (current > args.stop_f) {
      values.push_back(current);
      current += args.step_f;
    }
  }
  return values;
}

static std::pair<double, double> addmm_alpha_beta(
    const OrderedJson& arguments) {
  if (!arguments.is_array() || arguments.empty()) {
    return {1.0, 1.0};
  }
  if (arguments.size() < 2 ||
      !(json_is_numeric(arguments.at(0)) && json_is_numeric(arguments.at(1)))) {
    throw std::invalid_argument(
        "[ir.lowering] AddMM arguments must include [alpha, beta]");
  }
  return {arguments.at(0).get<double>(), arguments.at(1).get<double>()};
}

static bool sqrt_is_reciprocal(const OrderedJson& arguments) {
  if (!arguments.is_array() || arguments.empty()) {
    return false;
  }
  if (!arguments.at(0).is_boolean()) {
    throw std::invalid_argument(
        "[ir.lowering] Sqrt reciprocal flag must be boolean when present");
  }
  return arguments.at(0).get<bool>();
}

static AsStridedArguments asstrided_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 3)) {
    throw std::invalid_argument(
        "[ir.lowering] AsStrided arguments must include [shape, strides, offset]");
  }

  AsStridedArguments out;
  out.output_shape =
      normalize_integer_vector(arguments.at(0), "AsStrided shape");
  out.strides = normalize_integer_vector(arguments.at(1), "AsStrided strides");
  out.offset = normalized_integer_scalar(arguments.at(2), "AsStrided offset");

  if (out.output_shape.size() != out.strides.size()) {
    std::ostringstream msg;
    msg << "[ir.lowering] AsStrided shape/strides length mismatch: "
        << out.output_shape.size() << "/" << out.strides.size();
    throw std::invalid_argument(msg.str());
  }

  if (std::any_of(
          out.output_shape.begin(), out.output_shape.end(), [](int64_t value) {
            return value < 0;
          })) {
    throw std::invalid_argument(
        "[ir.lowering] AsStrided shape values must be non-negative");
  }

  return out;
}

static int64_t tensor_size_from_shape(const Shape& shape) {
  if (shape.empty()) {
    return 1;
  }
  return std::accumulate(
      shape.begin(),
      shape.end(),
      static_cast<int64_t>(1),
      std::multiplies<int64_t>());
}

static std::vector<int64_t> asstrided_linear_indices(
    const Shape& output_shape,
    const std::vector<int64_t>& strides,
    int64_t offset,
    int64_t input_size) {
  if (std::any_of(output_shape.begin(), output_shape.end(), [](int64_t value) {
        return value == 0;
      })) {
    return {};
  }

  int64_t total = output_shape.empty() ? 1
                                       : std::accumulate(
                                             output_shape.begin(),
                                             output_shape.end(),
                                             static_cast<int64_t>(1),
                                             std::multiplies<int64_t>());

  std::vector<int64_t> indices;
  indices.reserve(static_cast<size_t>(total));

  for (int64_t linear_index = 0; linear_index < total; ++linear_index) {
    int64_t remainder = linear_index;
    int64_t source_index = offset;

    for (int64_t axis = static_cast<int64_t>(output_shape.size()) - 1;
         axis >= 0;
         --axis) {
      const auto dim = output_shape[static_cast<size_t>(axis)];
      const auto coord = remainder % dim;
      remainder /= dim;
      source_index += coord * strides[static_cast<size_t>(axis)];
    }

    if (source_index < 0 || source_index >= input_size) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported AsStrided index " << source_index
          << " out of bounds for input size " << input_size;
      throw std::runtime_error(out.str());
    }
    indices.push_back(source_index);
  }

  return indices;
}

static std::pair<std::vector<std::string>, std::vector<OrderedJson>>
cast_inputs_to_dtype(
    size_t node_index,
    const std::string& op_name,
    const std::vector<std::string>& inputs,
    const std::string& target_dtype,
    ShapeMap& known_shapes,
    DtypeMap& known_dtypes,
    NameSet& used_tensor_names,
    const std::optional<std::vector<size_t>>& indices = std::nullopt) {
  // Inject Cast nodes for selected inputs and update known shape/dtype maps.
  std::vector<OrderedJson> cast_nodes;
  std::vector<std::string> casted_inputs = inputs;
  const auto& cast_to = onnx_dtype_symbol(target_dtype);

  std::set<size_t> index_filter;
  if (indices.has_value()) {
    for (const auto index : indices.value()) {
      index_filter.insert(index);
    }
  }

  for (size_t index = 0; index < inputs.size(); ++index) {
    if (indices.has_value() && index_filter.find(index) == index_filter.end()) {
      continue;
    }

    const auto& input_name = inputs[index];
    const auto input_dtype =
        canonical_dtype(known_dtype_for(known_dtypes, input_name));
    if (!input_dtype.has_value() || input_dtype.value() == target_dtype) {
      continue;
    }

    const auto cast_output = unique_aux_tensor_name(
        used_tensor_names,
        node_index,
        op_name + "_input" + std::to_string(index) + "_cast");

    cast_nodes.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_" + op_name + "CastInput" +
            std::to_string(index),
        "Cast",
        {input_name},
        {cast_output},
        OrderedJson::object({{"to", cast_to}})));

    const auto input_shape = known_shape_for(known_shapes, input_name);
    if (input_shape.has_value()) {
      known_shapes[cast_output] = input_shape.value();
    }
    known_dtypes[cast_output] = target_dtype;
    casted_inputs[index] = cast_output;
  }

  return {casted_inputs, cast_nodes};
}

static std::string append_aux_int64_initializer(
    OrderedJson& initializers,
    NameSet& used_tensor_names,
    size_t node_index,
    const std::string& label,
    const std::vector<int64_t>& values) {
  // Most ONNX control tensors (axes/shape/pads/steps) are int64 initializers.
  const auto name =
      unique_aux_tensor_name(used_tensor_names, node_index, label);

  OrderedJson tensor = OrderedJson::object();
  tensor["name"] = name;
  tensor["shape"] = json_from_int_vector({static_cast<int64_t>(values.size())});
  tensor["dtype"] = "int64";
  tensor["values"] = json_from_int_vector(values);

  initializers.push_back(onnx_initializer_info(tensor));
  return name;
}

static std::string append_aux_float_initializer(
    OrderedJson& initializers,
    NameSet& used_tensor_names,
    size_t node_index,
    const std::string& label,
    const std::vector<double>& values,
    const std::string& dtype = "float32") {
  const auto name =
      unique_aux_tensor_name(used_tensor_names, node_index, label);

  OrderedJson tensor = OrderedJson::object();
  tensor["name"] = name;
  tensor["shape"] = json_from_int_vector({static_cast<int64_t>(values.size())});
  tensor["dtype"] = dtype;
  tensor["values"] = json_from_double_vector(values);

  initializers.push_back(onnx_initializer_info(tensor));
  return name;
}

static std::optional<std::string> flatten_onnx_op_type(
    const OrderedJson& arguments,
    bool strict,
    const OrderedJson* node,
    const ShapeMap* known_shapes) {
  if (!(arguments.is_array() && arguments.size() == 2 &&
        json_is_numeric(arguments.at(0)) && json_is_numeric(arguments.at(1)))) {
    if (!strict) {
      return std::nullopt;
    }
    std::ostringstream out;
    out << "[ir.lowering] unsupported Flatten arguments "
        << arguments.dump() << "; expected [start_axis, end_axis]";
    throw std::runtime_error(out.str());
  }

  if (known_shapes != nullptr && node != nullptr) {
    const auto inputs =
        parse_string_array(node->at("inputs"), "Flatten inputs");
    if (inputs.empty()) {
      throw std::runtime_error(
          "[ir.lowering] Flatten requires one input");
    }
    const auto input_name = inputs.front();
    const auto input_shape = known_shape_for(*known_shapes, input_name);
    if (!input_shape.has_value()) {
      if (!strict) {
        return std::nullopt;
      }
      std::ostringstream out;
      out << "[ir.lowering] unsupported Flatten for tensor " << input_name
          << " without known static shape";
      throw std::runtime_error(out.str());
    }

    flatten_shape_from_arguments(arguments, input_shape.value());
  }

  return onnx_op_name("Flatten");
}

static std::optional<std::string> convolution_onnx_op_type(
    const OrderedJson& arguments,
    bool strict) {
  const auto parsed = convolution_attributes_from_arguments(arguments, strict);
  if (!parsed.has_value()) {
    return std::nullopt;
  }

  if (parsed->flip) {
    return onnx_op_name("ConvolutionTranspose");
  }

  if (std::any_of(
          parsed->input_dilation.begin(),
          parsed->input_dilation.end(),
          [](int64_t value) { return value != 1; })) {
    if (!strict) {
      return std::nullopt;
    }

    std::ostringstream out;
    out << "[ir.lowering] unsupported Convolution input_dilation "
        << json_from_int_vector(parsed->input_dilation).dump()
        << "; only all-ones input_dilation is supported for flip=false";
    throw std::runtime_error(out.str());
  }

  return onnx_op_name("Convolution");
}

std::optional<std::string> onnx_op_type_for_node(
    const OrderedJson& node,
    bool strict,
    const ShapeMap* known_shapes) {
  // "strict=false" is used by compatibility probes to avoid throwing on every
  // unsupported node and instead return nullopt when possible.
  const auto op = node.at("op").get<std::string>();
  const OrderedJson arguments =
      node.contains("arguments") ? node.at("arguments") : OrderedJson::array();

  if (op == "Convolution") {
    return convolution_onnx_op_type(arguments, strict);
  }
  if (op == "Reduce") {
    return reduce_onnx_op_type(arguments, strict);
  }
  if (op == "ArgReduce") {
    return argreduce_onnx_op_type(arguments, strict);
  }
  if (op == "Flatten") {
    return flatten_onnx_op_type(arguments, strict, &node, known_shapes);
  }
  if (op == "Concatenate") {
    if (!concatenate_axis_from_arguments(arguments, strict).has_value()) {
      return std::nullopt;
    }
    return onnx_op_name(op);
  }

  const auto mapped = onnx_op_type_for_ir_op(op);
  if (mapped.has_value()) {
    return mapped;
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[ir.lowering] unsupported op " << op;
  throw std::runtime_error(out.str());
}

struct ParsedLoweringNode {
  std::string op;
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  OrderedJson attributes;
  OrderedJson arguments;
};

static ParsedLoweringNode parse_lowering_node(
    const OrderedJson& node,
    const ShapeMap& known_shapes) {
  ParsedLoweringNode parsed;
  parsed.op = node.at("op").get<std::string>();
  parsed.op_type = onnx_op_type_for_node(node, true, &known_shapes).value();
  parsed.inputs = parse_string_array(node.at("inputs"), "node inputs");
  parsed.outputs = parse_string_array(node.at("outputs"), "node outputs");
  parsed.attributes = onnx_node_attributes(node);
  parsed.arguments =
      node.contains("arguments") ? node.at("arguments") : OrderedJson::array();
  return parsed;
}

static void assign_known_shape_if_present(
    ShapeMap& known_shapes,
    const std::vector<std::string>& names,
    const std::optional<Shape>& shape) {
  if (!shape.has_value()) {
    return;
  }
  for (const auto& name : names) {
    known_shapes[name] = shape.value();
  }
}

static void assign_known_dtype_if_present(
    DtypeMap& known_dtypes,
    const std::vector<std::string>& names,
    const std::optional<std::string>& dtype) {
  if (!dtype.has_value()) {
    return;
  }
  for (const auto& name : names) {
    known_dtypes[name] = dtype.value();
  }
}

static std::optional<std::vector<OrderedJson>> maybe_lower_with_promoted_cast(
    size_t node_index,
    const std::string& op,
    const std::string& op_type,
    std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const OrderedJson& attributes,
    const std::optional<std::string>& promoted_dtype,
    const std::optional<std::vector<size_t>>& cast_indices,
    size_t lhs_input_index,
    size_t rhs_input_index,
    const std::optional<std::string>& output_dtype_override,
    LoweringContext& lowering) {
  // Shared lowering path for binary ops that require explicit dtype alignment.
  if (!promoted_dtype.has_value()) {
    return std::nullopt;
  }

  auto [casted_inputs, cast_nodes] = cast_inputs_to_dtype(
      node_index,
      op,
      inputs,
      promoted_dtype.value(),
      lowering.known_shapes,
      lowering.known_dtypes,
      lowering.used_tensor_names,
      cast_indices);
  if (cast_nodes.empty()) {
    return std::nullopt;
  }

  inputs = casted_inputs;
  const auto inferred_output_shape = infer_elementwise_output_shape(
      known_shape_for(lowering.known_shapes, inputs.at(lhs_input_index)),
      known_shape_for(lowering.known_shapes, inputs.at(rhs_input_index)));
  const auto inferred_output_dtype = output_dtype_override.has_value()
      ? output_dtype_override
      : promoted_dtype;
  assign_known_shape_if_present(
      lowering.known_shapes, outputs, inferred_output_shape);
  assign_known_dtype_if_present(
      lowering.known_dtypes, outputs, inferred_output_dtype);

  cast_nodes.push_back(build_onnx_node_spec(
      "node_" + std::to_string(node_index) + "_" + op_type,
      op_type,
      inputs,
      outputs,
      attributes));
  return cast_nodes;
}

static Shape require_known_static_shape_for_op(
    const ShapeMap& known_shapes,
    const std::string& op,
    const std::string& role,
    const std::string& tensor_name) {
  // Some rewrites need concrete shapes (e.g. Gather/GatherAxis/AsStrided).
  const auto shape = known_shape_for(known_shapes, tensor_name);
  if (shape.has_value()) {
    return shape.value();
  }

  std::ostringstream out;
  out << "[ir.lowering] unsupported " << op << " for " << role << " "
      << tensor_name << " without known static shape";
  throw std::runtime_error(out.str());
}

static void infer_shape_dtype_from_input_tensor(
    const std::string& input_name,
    const ShapeMap& known_shapes,
    const DtypeMap& known_dtypes,
    std::optional<Shape>& inferred_output_shape,
    std::optional<std::string>& inferred_output_dtype) {
  inferred_output_shape = known_shape_for(known_shapes, input_name);
  inferred_output_dtype = known_dtype_for(known_dtypes, input_name);
}

static void ensure_indices_input_is_int64(
    size_t node_index,
    const std::string& cast_node_suffix,
    const std::string& cast_aux_label,
    std::vector<std::string>& inputs,
    size_t indices_input_position,
    std::vector<OrderedJson>& pre_nodes,
    LoweringContext& lowering) {
  // ONNX Gather/GatherElements consume integer indices; normalize to int64.
  auto& known_shapes = lowering.known_shapes;
  auto& known_dtypes = lowering.known_dtypes;
  auto& used_tensor_names = lowering.used_tensor_names;

  auto indices_input = inputs.at(indices_input_position);
  const auto indices_dtype =
      canonical_dtype(known_dtype_for(known_dtypes, indices_input));
  if (!indices_dtype.has_value() || indices_dtype.value() == "int32" ||
      indices_dtype.value() == "int64") {
    return;
  }

  const auto cast_indices =
      unique_aux_tensor_name(used_tensor_names, node_index, cast_aux_label);
  pre_nodes.push_back(build_onnx_node_spec(
      "node_" + std::to_string(node_index) + "_" + cast_node_suffix,
      "Cast",
      {indices_input},
      {cast_indices},
      OrderedJson::object({{"to", onnx_dtype_symbol("int64")}})));

  const auto index_shape = known_shape_for(known_shapes, indices_input);
  if (index_shape.has_value()) {
    known_shapes[cast_indices] = index_shape.value();
  }
  known_dtypes[cast_indices] = "int64";
  inputs[indices_input_position] = cast_indices;
}

static bool is_passthrough_unary_op(const std::string& op) {
  return op == "Exp" || op == "Log" || op == "Abs" || op == "Negative" ||
      op == "Relu" || op == "Sigmoid" || op == "Tanh" || op == "Softmax" ||
      op == "Sin" || op == "Cos" || op == "Erf" || op == "Floor";
}

static std::vector<OrderedJson> lower_onnx_arange_node(
    const std::vector<std::string>& outputs,
    const OrderedJson& arguments,
    LoweringContext& lowering) {
  // Lower Arange to a static initializer to keep ONNX graph deterministic.
  const auto parsed = arange_arguments(arguments);
  const auto values = arange_values(parsed);

  OrderedJson tensor = OrderedJson::object();
  tensor["name"] = outputs.at(0);
  tensor["shape"] = json_from_int_vector({static_cast<int64_t>(values.size())});
  tensor["dtype"] = parsed.dtype;
  tensor["values"] = values;

  lowering.initializers.push_back(onnx_initializer_info(tensor));
  lowering.known_shapes[outputs.at(0)] = {static_cast<int64_t>(values.size())};
  lowering.known_dtypes[outputs.at(0)] = parsed.dtype;
  return {};
}

static std::vector<OrderedJson> lower_onnx_convolution_node(
    size_t node_index,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const OrderedJson& arguments,
    LoweringContext& lowering) {
  // Both Convolution and ConvolutionTranspose are expressed through layout
  // transposes because IR tensor layout differs from ONNX Conv kernels.
  auto& used_tensor_names = lowering.used_tensor_names;
  auto& known_shapes = lowering.known_shapes;
  auto& known_dtypes = lowering.known_dtypes;
  const auto convolution =
      convolution_attributes_from_arguments(arguments, true).value();

  if (convolution.flip) {
    // flip=true is emitted as ConvTranspose with derived paddings.
    const auto spatial_rank = convolution.spatial_rank;
    const auto [input_perm, output_perm] =
        convolution_data_permutations(spatial_rank);
    const auto weight_perm =
        convolution_transpose_weight_permutation(spatial_rank);

    const auto input_shape = known_shape_for(known_shapes, inputs[0]);
    const auto weight_shape = known_shape_for(known_shapes, inputs[1]);
    if (!weight_shape.has_value()) {
      throw std::runtime_error(
          "[ir.lowering] unsupported Convolution flip=true without known static weight shape");
    }

    const auto transposed_input = unique_aux_tensor_name(
        used_tensor_names, node_index, "conv_transpose_input_ncx");
    const auto transposed_weight = unique_aux_tensor_name(
        used_tensor_names, node_index, "conv_transpose_weight_icx");
    const auto conv_output = unique_aux_tensor_name(
        used_tensor_names, node_index, "conv_transpose_output_ncx");

    if (input_shape.has_value()) {
      known_shapes[transposed_input] = permute_shape(
          input_shape.value(),
          input_perm,
          "ConvolutionTranspose input permutation");
    }
    known_shapes[transposed_weight] = permute_shape(
        weight_shape.value(),
        weight_perm,
        "ConvolutionTranspose weight permutation");

    const auto conv_transpose = convtranspose_attributes_from_convolution(
        convolution, weight_shape.value());
    auto inferred_output_shape = infer_convolution_transpose_output_shape(
        input_shape,
        weight_shape,
        conv_transpose.strides,
        conv_transpose.pads_begin,
        conv_transpose.pads_end,
        conv_transpose.dilations,
        conv_transpose.output_padding,
        convolution.groups);
    if (inferred_output_shape.has_value()) {
      known_shapes[conv_output] = permute_shape(
          inferred_output_shape.value(),
          input_perm,
          "ConvolutionTranspose output permutation");
      for (const auto& name : outputs) {
        known_shapes[name] = inferred_output_shape.value();
      }
    }

    OrderedJson conv_transpose_attributes = OrderedJson::object();
    conv_transpose_attributes["strides"] =
        json_from_int_vector(conv_transpose.strides);
    conv_transpose_attributes["pads"] =
        json_from_int_vector(conv_transpose.pads);
    conv_transpose_attributes["dilations"] =
        json_from_int_vector(conv_transpose.dilations);
    conv_transpose_attributes["group"] = convolution.groups;
    conv_transpose_attributes["output_padding"] =
        json_from_int_vector(conv_transpose.output_padding);

    const auto input_dtype = known_dtype_for(known_dtypes, inputs[0]);
    const auto weight_dtype = known_dtype_for(known_dtypes, inputs[1]);
    auto conv_output_dtype = promote_binary_dtype(input_dtype, weight_dtype);
    if (!conv_output_dtype.has_value()) {
      conv_output_dtype = input_dtype.has_value() ? input_dtype : weight_dtype;
    }

    if (input_dtype.has_value()) {
      known_dtypes[transposed_input] = input_dtype.value();
    }
    if (weight_dtype.has_value()) {
      known_dtypes[transposed_weight] = weight_dtype.value();
    }
    if (conv_output_dtype.has_value()) {
      known_dtypes[conv_output] = conv_output_dtype.value();
      for (const auto& name : outputs) {
        known_dtypes[name] = conv_output_dtype.value();
      }
    }

    return {
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_InputTranspose",
            "Transpose",
            {inputs[0]},
            {transposed_input},
            OrderedJson::object({{"perm", json_from_int_vector(input_perm)}})),
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_WeightTranspose",
            "Transpose",
            {inputs[1]},
            {transposed_weight},
            OrderedJson::object({{"perm", json_from_int_vector(weight_perm)}})),
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_ConvTranspose",
            "ConvTranspose",
            {transposed_input, transposed_weight},
            {conv_output},
            conv_transpose_attributes),
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_OutputTranspose",
            "Transpose",
            {conv_output},
            outputs,
            OrderedJson::object(
                {{"perm", json_from_int_vector(output_perm)}}))};
  }

  if (std::any_of(
          convolution.input_dilation.begin(),
          convolution.input_dilation.end(),
          [](int64_t value) { return value != 1; })) {
    std::ostringstream out;
    out << "[ir.lowering] unsupported Convolution input_dilation "
        << json_from_int_vector(convolution.input_dilation).dump()
        << "; only all-ones input_dilation is supported for flip=false";
    throw std::runtime_error(out.str());
  }

  const auto spatial_rank = convolution.spatial_rank;
  const auto [input_perm, output_perm] =
      convolution_data_permutations(spatial_rank);
  const auto weight_perm = convolution_weight_permutation(spatial_rank);

  const auto input_shape = known_shape_for(known_shapes, inputs[0]);
  const auto weight_shape = known_shape_for(known_shapes, inputs[1]);
  const size_t input_rank = spatial_rank + 2;

  if (input_shape.has_value() && input_shape->size() != input_rank) {
    std::ostringstream out;
    out << "[ir.lowering] Convolution input rank mismatch: expected "
        << input_rank << ", got " << input_shape->size();
    throw std::invalid_argument(out.str());
  }
  if (weight_shape.has_value() && weight_shape->size() != input_rank) {
    std::ostringstream out;
    out << "[ir.lowering] Convolution weight rank mismatch: expected "
        << input_rank << ", got " << weight_shape->size();
    throw std::invalid_argument(out.str());
  }

  const auto transposed_input =
      unique_aux_tensor_name(used_tensor_names, node_index, "conv_input_ncx");
  const auto transposed_weight =
      unique_aux_tensor_name(used_tensor_names, node_index, "conv_weight_ocx");
  const auto conv_output =
      unique_aux_tensor_name(used_tensor_names, node_index, "conv_output_ncx");

  if (input_shape.has_value()) {
    known_shapes[transposed_input] = permute_shape(
        input_shape.value(), input_perm, "Convolution input permutation");
  }
  if (weight_shape.has_value()) {
    known_shapes[transposed_weight] = permute_shape(
        weight_shape.value(), weight_perm, "Convolution weight permutation");
  }

  auto inferred_output_shape = infer_convolution_output_shape(
      input_shape,
      weight_shape,
      convolution.strides,
      convolution.padding_low,
      convolution.padding_high,
      convolution.kernel_dilation,
      convolution.groups);
  if (inferred_output_shape.has_value()) {
    known_shapes[conv_output] = permute_shape(
        inferred_output_shape.value(),
        input_perm,
        "Convolution output permutation");
    for (const auto& name : outputs) {
      known_shapes[name] = inferred_output_shape.value();
    }
  }

  OrderedJson conv_attributes = OrderedJson::object();
  conv_attributes["strides"] = json_from_int_vector(convolution.strides);
  conv_attributes["pads"] = json_from_int_vector(convolution.pads);
  conv_attributes["dilations"] =
      json_from_int_vector(convolution.kernel_dilation);
  conv_attributes["group"] = convolution.groups;

  return {
      build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_InputTranspose",
          "Transpose",
          {inputs[0]},
          {transposed_input},
          OrderedJson::object({{"perm", json_from_int_vector(input_perm)}})),
      build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_WeightTranspose",
          "Transpose",
          {inputs[1]},
          {transposed_weight},
          OrderedJson::object({{"perm", json_from_int_vector(weight_perm)}})),
      build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_Conv",
          "Conv",
          {transposed_input, transposed_weight},
          {conv_output},
          conv_attributes),
      build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_OutputTranspose",
          "Transpose",
          {conv_output},
          outputs,
          OrderedJson::object({{"perm", json_from_int_vector(output_perm)}}))};
}

std::vector<OrderedJson> lower_onnx_node_default(
    const OrderedJson& node,
    size_t node_index,
    LoweringContext& lowering) {
  // Main per-node lowering pipeline:
  // 1) Parse/match op and arguments.
  // 2) Optionally inject helper initializers/nodes.
  // 3) Track best-effort output shape/dtype facts.
  // 4) Return one or more ONNX nodes for this source node.
  auto& initializers = lowering.initializers;
  auto& used_tensor_names = lowering.used_tensor_names;
  auto& known_shapes = lowering.known_shapes;
  auto& known_dtypes = lowering.known_dtypes;

  auto parsed_node = parse_lowering_node(node, known_shapes);
  const auto op = std::move(parsed_node.op);
  const auto op_type = std::move(parsed_node.op_type);
  auto inputs = std::move(parsed_node.inputs);
  auto outputs = std::move(parsed_node.outputs);
  OrderedJson attributes = std::move(parsed_node.attributes);
  const OrderedJson arguments = std::move(parsed_node.arguments);

  std::optional<Shape> inferred_output_shape;
  std::optional<std::string> inferred_output_dtype;

  if (op == "Arange") {
    return lower_onnx_arange_node(outputs, arguments, lowering);
  } else if (op == "RandomBits") {
    // IR RandomBits is stateful via explicit RNG key input.
    // ONNX RandomUniform is stateless, so we drop the key input and keep
    // shape/dtype/range semantics.
    const auto parsed = random_bits_arguments(arguments);
    inputs.clear();
    attributes["shape"] = json_from_shape(parsed.shape);
    attributes["dtype"] =
        onnx_elem_type_from_symbol_lookup(onnx_dtype_symbol(parsed.dtype));
    attributes["low"] = parsed.low;
    attributes["high"] = parsed.high;
    inferred_output_shape = parsed.shape;
    inferred_output_dtype = parsed.dtype;
  } else if (op == "ErfInv") {
    if (inputs.size() != 1 || outputs.size() != 1) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported ErfInv arity: inputs=" << inputs.size()
          << ", outputs=" << outputs.size();
      throw std::runtime_error(out.str());
    }

    auto input_name = inputs.front();
    const auto output_name = outputs.front();

    const auto input_dtype =
        canonical_dtype(known_dtype_for(known_dtypes, input_name));
    const bool cast_input_to_float32 =
        input_dtype.has_value() && input_dtype.value() != "float32";

    std::vector<OrderedJson> lowered;
    if (cast_input_to_float32) {
      const auto cast_input =
          unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_input_cast");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_ErfInvCastInput",
          "Cast",
          {input_name},
          {cast_input},
          OrderedJson::object({{"to", "FLOAT"}})));

      const auto input_shape = known_shape_for(known_shapes, input_name);
      if (input_shape.has_value()) {
        known_shapes[cast_input] = input_shape.value();
      }
      known_dtypes[cast_input] = "float32";
      input_name = cast_input;
    }

    constexpr double kErfInvA = 0.147;
    constexpr double kPi = 3.14159265358979323846;
    const double c = 2.0 / (kPi * kErfInvA);

    const auto one_name = append_aux_float_initializer(
        initializers, used_tensor_names, node_index, "erfinv_one", {1.0});
    const auto two_name = append_aux_float_initializer(
        initializers, used_tensor_names, node_index, "erfinv_two", {2.0});
    const auto zero_name = append_aux_float_initializer(
        initializers, used_tensor_names, node_index, "erfinv_zero", {0.0});
    const auto a_name = append_aux_float_initializer(
        initializers, used_tensor_names, node_index, "erfinv_a", {kErfInvA});
    const auto c_name = append_aux_float_initializer(
        initializers, used_tensor_names, node_index, "erfinv_c", {c});

    const auto abs_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_abs");
    const auto square_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_square");
    const auto one_minus_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_one_minus");
    const auto log_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_log");
    const auto log_half_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_log_half");
    const auto base_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_base");
    const auto base_square_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_base_square");
    const auto log_div_a_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_log_div_a");
    const auto inner_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_inner");
    const auto sqrt_inner_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_sqrt_inner");
    const auto magnitude_square_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "erfinv_magnitude_square");
    const auto magnitude_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_magnitude");
    const auto less_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_less");
    const auto neg_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_neg");
    const auto where_out =
        unique_aux_tensor_name(used_tensor_names, node_index, "erfinv_where");

    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvAbs",
        "Abs",
        {input_name},
        {abs_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvSquare",
        "Mul",
        {abs_out, abs_out},
        {square_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvOneMinus",
        "Sub",
        {one_name, square_out},
        {one_minus_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvLog",
        "Log",
        {one_minus_out},
        {log_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvLogHalf",
        "Div",
        {log_out, two_name},
        {log_half_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvBase",
        "Add",
        {c_name, log_half_out},
        {base_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvBaseSquare",
        "Mul",
        {base_out, base_out},
        {base_square_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvLogDivA",
        "Div",
        {log_out, a_name},
        {log_div_a_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvInner",
        "Sub",
        {base_square_out, log_div_a_out},
        {inner_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvSqrtInner",
        "Sqrt",
        {inner_out},
        {sqrt_inner_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvMagnitudeSquare",
        "Sub",
        {sqrt_inner_out, base_out},
        {magnitude_square_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvMagnitude",
        "Sqrt",
        {magnitude_square_out},
        {magnitude_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvLessZero",
        "Less",
        {input_name, zero_name},
        {less_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvNeg",
        "Neg",
        {magnitude_out},
        {neg_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_ErfInvWhere",
        "Where",
        {less_out, neg_out, magnitude_out},
        {where_out},
        OrderedJson::object()));

    if (cast_input_to_float32) {
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_ErfInvCastOutput",
          "Cast",
          {where_out},
          outputs,
          OrderedJson::object(
              {{"to", onnx_dtype_symbol(input_dtype.value())}})));
      inferred_output_dtype = input_dtype.value();
    } else {
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_ErfInvOutput",
          "Identity",
          {where_out},
          outputs,
          OrderedJson::object()));
      inferred_output_dtype = "float32";
    }

    inferred_output_shape = known_shape_for(known_shapes, inputs.front());
    if (inferred_output_shape.has_value()) {
      known_shapes[output_name] = inferred_output_shape.value();
    }
    if (inferred_output_dtype.has_value()) {
      known_dtypes[output_name] = inferred_output_dtype.value();
    }
    return lowered;
  } else if (op == "LayerNorm") {
    if (inputs.empty() || inputs.size() > 3) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported LayerNorm input arity "
          << inputs.size() << "; expected 1..3";
      throw std::runtime_error(out.str());
    }

    const auto parsed = layernorm_arguments(arguments);
    const auto axis_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "layernorm_axes", {-1});
    const auto epsilon_name = append_aux_float_initializer(
        initializers,
        used_tensor_names,
        node_index,
        "layernorm_epsilon",
        {parsed.epsilon});

    std::vector<OrderedJson> lowered;
    const auto input_name = inputs.front();
    const auto input_dtype =
        canonical_dtype(known_dtype_for(known_dtypes, input_name));
    auto normalized_input = input_name;

    if (input_dtype.has_value() && input_dtype.value() != "float32") {
      normalized_input = unique_aux_tensor_name(
          used_tensor_names, node_index, "layernorm_input_float");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_LayerNormCastInput",
          "Cast",
          {input_name},
          {normalized_input},
          OrderedJson::object({{"to", onnx_dtype_symbol("float32")}})));
      if (const auto input_shape = known_shape_for(known_shapes, input_name);
          input_shape.has_value()) {
        known_shapes[normalized_input] = input_shape.value();
      }
      known_dtypes[normalized_input] = "float32";
    }

    const auto mean_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "layernorm_mean");
    const auto centered_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "layernorm_centered");
    const auto square_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "layernorm_square");
    const auto var_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "layernorm_var");
    const auto var_eps_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "layernorm_var_eps");
    const auto std_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "layernorm_std");
    const auto inv_std_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "layernorm_inv_std");
    const auto normalized_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "layernorm_normalized");

    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_LayerNormMean",
        "ReduceMean",
        {normalized_input, axis_name},
        {mean_out},
        OrderedJson::object({{"keepdims", 1}})));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_LayerNormCentered",
        "Sub",
        {normalized_input, mean_out},
        {centered_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_LayerNormSquare",
        "Mul",
        {centered_out, centered_out},
        {square_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_LayerNormVar",
        "ReduceMean",
        {square_out, axis_name},
        {var_out},
        OrderedJson::object({{"keepdims", 1}})));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_LayerNormVarEps",
        "Add",
        {var_out, epsilon_name},
        {var_eps_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_LayerNormStd",
        "Sqrt",
        {var_eps_out},
        {std_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_LayerNormInvStd",
        "Reciprocal",
        {std_out},
        {inv_std_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_LayerNormNormalized",
        "Mul",
        {centered_out, inv_std_out},
        {normalized_out},
        OrderedJson::object()));

    if (const auto input_shape = known_shape_for(known_shapes, input_name);
        input_shape.has_value()) {
      known_shapes[normalized_out] = input_shape.value();
    }
    known_dtypes[normalized_out] = "float32";

    std::string affine_input = normalized_out;
    if (input_dtype.has_value() && input_dtype.value() != "float32") {
      const auto cast_back = unique_aux_tensor_name(
          used_tensor_names, node_index, "layernorm_cast_back");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_LayerNormCastBack",
          "Cast",
          {normalized_out},
          {cast_back},
          OrderedJson::object(
              {{"to", onnx_dtype_symbol(input_dtype.value())}})));
      if (const auto input_shape = known_shape_for(known_shapes, input_name);
          input_shape.has_value()) {
        known_shapes[cast_back] = input_shape.value();
      }
      known_dtypes[cast_back] = input_dtype.value();
      affine_input = cast_back;
    }

    std::string output_value = affine_input;
    if (inputs.size() >= 2) {
      const auto weighted = unique_aux_tensor_name(
          used_tensor_names, node_index, "layernorm_weighted");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_LayerNormWeight",
          "Mul",
          {output_value, inputs[1]},
          {weighted},
          OrderedJson::object()));
      output_value = weighted;
      if (const auto input_shape = known_shape_for(known_shapes, input_name);
          input_shape.has_value()) {
        known_shapes[weighted] = input_shape.value();
      }
      if (const auto weight_dtype = known_dtype_for(known_dtypes, inputs[1]);
          weight_dtype.has_value()) {
        known_dtypes[weighted] = weight_dtype.value();
      }
    }

    if (inputs.size() == 3) {
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_LayerNormBias",
          "Add",
          {output_value, inputs[2]},
          outputs,
          OrderedJson::object()));
    } else if (output_value != outputs.front() || outputs.size() != 1) {
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_LayerNormOutput",
          "Identity",
          {output_value},
          outputs,
          OrderedJson::object()));
    }

    inferred_output_shape = known_shape_for(known_shapes, input_name);
    inferred_output_dtype = input_dtype.has_value()
        ? input_dtype
        : known_dtype_for(known_dtypes, output_value);

    assign_known_shape_if_present(known_shapes, outputs, inferred_output_shape);
    assign_known_dtype_if_present(known_dtypes, outputs, inferred_output_dtype);
    return lowered;
  } else if (op == "RoPE") {
    if (!(inputs.size() == 2 || inputs.size() == 3)) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported RoPE input arity " << inputs.size()
          << "; expected 2 or 3";
      throw std::runtime_error(out.str());
    }

    const auto parsed = rope_arguments(arguments);
    const auto input_name = inputs.front();
    const Shape input_shape = require_known_static_shape_for_op(
        known_shapes, "RoPE", "tensor", input_name);
    if (input_shape.size() < 3 || input_shape.size() > 4) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported RoPE input rank " << input_shape.size()
          << "; expected rank 3 or 4";
      throw std::runtime_error(out.str());
    }

    Shape work_shape = input_shape;
    std::vector<OrderedJson> lowered;
    std::string work_input = input_name;
    if (input_shape.size() == 3) {
      const auto axes_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "rope_expand_axes",
          {1});
      work_input = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_expanded_input");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEExpandInput",
          "Unsqueeze",
          {input_name, axes_name},
          {work_input},
          OrderedJson::object()));
      work_shape = {input_shape[0], 1, input_shape[1], input_shape[2]};
      known_shapes[work_input] = work_shape;
      if (const auto input_dtype = known_dtype_for(known_dtypes, input_name);
          input_dtype.has_value()) {
        known_dtypes[work_input] = input_dtype.value();
      }
    }

    if (work_shape.size() != 4) {
      throw std::runtime_error(
          "[ir.lowering] unsupported RoPE internal rank after normalization");
    }

    const auto channels = work_shape[3];
    const bool channels_known = channels > 0;
    if (channels == 0 || channels < -1) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported RoPE channels " << channels
          << "; expected positive channel dimension or -1 unknown";
      throw std::runtime_error(out.str());
    }
    if (channels_known && parsed.dims > channels) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported RoPE dims " << parsed.dims
          << "; exceeds channel dimension " << channels;
      throw std::runtime_error(out.str());
    }

    const auto seq_len = work_shape[2];
    if (seq_len <= 0) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported RoPE sequence length " << seq_len
          << "; expected positive static sequence length";
      throw std::runtime_error(out.str());
    }

    const auto half_dims = parsed.dims / 2;
    const auto input_dtype =
        canonical_dtype(known_dtype_for(known_dtypes, input_name));

    const auto offset_shape = known_shape_for(known_shapes, inputs[1]);
    std::string offset_input = inputs[1];
    if (const auto offset_dtype = canonical_dtype(
            known_dtype_for(known_dtypes, offset_input));
        !offset_dtype.has_value() || offset_dtype.value() != "float32") {
      const auto offset_cast = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_offset_float");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPECastOffset",
          "Cast",
          {offset_input},
          {offset_cast},
          OrderedJson::object({{"to", onnx_dtype_symbol("float32")}})));
      if (offset_shape.has_value()) {
        known_shapes[offset_cast] = offset_shape.value();
      }
      known_dtypes[offset_cast] = "float32";
      offset_input = offset_cast;
    }

    if (offset_shape.has_value() && offset_shape->size() == 1) {
      if (offset_shape->at(0) != work_shape[0]) {
        std::ostringstream out;
        out << "[ir.lowering] unsupported RoPE offset shape "
            << json_from_shape(offset_shape.value()).dump()
            << "; expected scalar or vector with " << work_shape[0]
            << " elements";
        throw std::runtime_error(out.str());
      }
      const auto axes_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "rope_offset_expand_axes",
          {1, 2});
      const auto offset_expanded = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_offset_expanded");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEExpandOffset",
          "Unsqueeze",
          {offset_input, axes_name},
          {offset_expanded},
          OrderedJson::object()));
      offset_input = offset_expanded;
      known_shapes[offset_expanded] = {work_shape[0], 1, 1};
      known_dtypes[offset_expanded] = "float32";
    } else if (offset_shape.has_value() && !offset_shape->empty()) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported RoPE offset rank "
          << offset_shape->size()
          << "; expected scalar or 1D vector";
      throw std::runtime_error(out.str());
    }

    std::vector<double> base_positions(static_cast<size_t>(seq_len), 0.0);
    for (int64_t i = 0; i < seq_len; ++i) {
      base_positions[static_cast<size_t>(i)] = static_cast<double>(i);
    }
    const auto position_base_name = append_aux_float_initializer(
        initializers,
        used_tensor_names,
        node_index,
        "rope_positions_base",
        base_positions);
    const auto scale_name = append_aux_float_initializer(
        initializers,
        used_tensor_names,
        node_index,
        "rope_scale",
        {parsed.scale});

    const auto positions_offset = unique_aux_tensor_name(
        used_tensor_names, node_index, "rope_positions_offset");
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_RoPEAddOffset",
        "Add",
        {position_base_name, offset_input},
        {positions_offset},
        OrderedJson::object()));

    const auto positions_scaled = unique_aux_tensor_name(
        used_tensor_names, node_index, "rope_positions");
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_RoPEScale",
        "Mul",
        {positions_offset, scale_name},
        {positions_scaled},
        OrderedJson::object()));
    known_dtypes[positions_scaled] = "float32";

    const auto positions_shape = (offset_shape.has_value() &&
                                  offset_shape->size() == 1)
        ? std::optional<Shape>({work_shape[0], 1, seq_len})
        : std::optional<Shape>({seq_len});
    if (positions_shape.has_value()) {
      known_shapes[positions_scaled] = positions_shape.value();
    }

    std::string inv_freqs_input;
    if (inputs.size() == 3) {
      inv_freqs_input = inputs[2];
      const auto freqs_dtype = canonical_dtype(
          known_dtype_for(known_dtypes, inv_freqs_input));
      if (!freqs_dtype.has_value() || freqs_dtype.value() != "float32") {
        const auto cast_freqs = unique_aux_tensor_name(
            used_tensor_names, node_index, "rope_freqs_float");
        lowered.push_back(build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_RoPECastFreqs",
            "Cast",
            {inv_freqs_input},
            {cast_freqs},
            OrderedJson::object({{"to", onnx_dtype_symbol("float32")}})));
        if (const auto freqs_shape =
                known_shape_for(known_shapes, inv_freqs_input);
            freqs_shape.has_value()) {
          known_shapes[cast_freqs] = freqs_shape.value();
        }
        known_dtypes[cast_freqs] = "float32";
        inv_freqs_input = cast_freqs;
      }

      const auto reciprocal_out = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_inv_freqs");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEReciprocalFreqs",
          "Reciprocal",
          {inv_freqs_input},
          {reciprocal_out},
          OrderedJson::object()));
      inv_freqs_input = reciprocal_out;
      if (const auto freqs_shape = known_shape_for(known_shapes, inputs[2]);
          freqs_shape.has_value()) {
        known_shapes[inv_freqs_input] = freqs_shape.value();
      }
      known_dtypes[inv_freqs_input] = "float32";
    } else {
      std::vector<double> inv_freqs_values(
          static_cast<size_t>(half_dims), 0.0);
      const double base_log = std::log(parsed.base);
      for (int64_t i = 0; i < half_dims; ++i) {
        inv_freqs_values[static_cast<size_t>(i)] =
            std::exp((static_cast<double>(-i) * base_log) /
                     static_cast<double>(half_dims));
      }
      inv_freqs_input = append_aux_float_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "rope_inv_freqs",
          inv_freqs_values);
    }

    const auto positions_rank = positions_shape.has_value()
        ? static_cast<int64_t>(positions_shape->size())
        : 1;
    const auto position_unsqueeze_axes = append_aux_int64_initializer(
        initializers,
        used_tensor_names,
        node_index,
        "rope_position_unsqueeze_axes",
        {positions_rank});
    const auto positions_unsqueezed = unique_aux_tensor_name(
        used_tensor_names, node_index, "rope_positions_unsqueezed");
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_RoPEUnsqueezePositions",
        "Unsqueeze",
        {positions_scaled, position_unsqueeze_axes},
        {positions_unsqueezed},
        OrderedJson::object()));
    if (positions_shape.has_value()) {
      auto shape = positions_shape.value();
      shape.push_back(1);
      known_shapes[positions_unsqueezed] = shape;
    }
    known_dtypes[positions_unsqueezed] = "float32";

    const auto theta_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "rope_theta");
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_RoPETheta",
        "Mul",
        {positions_unsqueezed, inv_freqs_input},
        {theta_out},
        OrderedJson::object()));
    known_dtypes[theta_out] = "float32";

    std::string cos_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "rope_cos");
    std::string sin_out = unique_aux_tensor_name(
        used_tensor_names, node_index, "rope_sin");
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_RoPECos",
        "Cos",
        {theta_out},
        {cos_out},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_RoPESin",
        "Sin",
        {theta_out},
        {sin_out},
        OrderedJson::object()));
    known_dtypes[cos_out] = "float32";
    known_dtypes[sin_out] = "float32";

    if (input_dtype.has_value() && input_dtype.value() != "float32") {
      const auto cos_cast = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_cos_cast");
      const auto sin_cast = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_sin_cast");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPECastCos",
          "Cast",
          {cos_out},
          {cos_cast},
          OrderedJson::object(
              {{"to", onnx_dtype_symbol(input_dtype.value())}})));
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPECastSin",
          "Cast",
          {sin_out},
          {sin_cast},
          OrderedJson::object(
              {{"to", onnx_dtype_symbol(input_dtype.value())}})));
      cos_out = cos_cast;
      sin_out = sin_cast;
      known_dtypes[cos_out] = input_dtype.value();
      known_dtypes[sin_out] = input_dtype.value();
    }

    const auto axis_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "rope_slice_axis", {3});

    auto build_slice = [&](const std::string& output_name,
                           int64_t start,
                           int64_t end,
                           int64_t step) {
      const auto starts_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "rope_slice_starts",
          {start});
      const auto ends_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "rope_slice_ends",
          {end});
      const auto steps_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "rope_slice_steps",
          {step});
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPESlice_" + output_name,
          "Slice",
          {work_input, starts_name, ends_name, axis_name, steps_name},
          {output_name},
          OrderedJson::object()));
    };

    const auto x1_name =
        unique_aux_tensor_name(used_tensor_names, node_index, "rope_x1");
    const auto x2_name =
        unique_aux_tensor_name(used_tensor_names, node_index, "rope_x2");
    if (parsed.traditional) {
      build_slice(x1_name, 0, parsed.dims, 2);
      build_slice(x2_name, 1, parsed.dims, 2);
    } else {
      build_slice(x1_name, 0, half_dims, 1);
      build_slice(x2_name, half_dims, parsed.dims, 1);
    }

    const Shape pair_shape = {work_shape[0], work_shape[1], work_shape[2], half_dims};
    known_shapes[x1_name] = pair_shape;
    known_shapes[x2_name] = pair_shape;
    if (input_dtype.has_value()) {
      known_dtypes[x1_name] = input_dtype.value();
      known_dtypes[x2_name] = input_dtype.value();
    }

    const auto x1_cos = unique_aux_tensor_name(
        used_tensor_names, node_index, "rope_x1_cos");
    const auto x2_sin = unique_aux_tensor_name(
        used_tensor_names, node_index, "rope_x2_sin");
    const auto x1_sin = unique_aux_tensor_name(
        used_tensor_names, node_index, "rope_x1_sin");
    const auto x2_cos = unique_aux_tensor_name(
        used_tensor_names, node_index, "rope_x2_cos");
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_RoPEMulX1Cos",
        "Mul",
        {x1_name, cos_out},
        {x1_cos},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_RoPEMulX2Sin",
        "Mul",
        {x2_name, sin_out},
        {x2_sin},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_RoPEMulX1Sin",
        "Mul",
        {x1_name, sin_out},
        {x1_sin},
        OrderedJson::object()));
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_RoPEMulX2Cos",
        "Mul",
        {x2_name, cos_out},
        {x2_cos},
        OrderedJson::object()));

    const auto y1_name =
        unique_aux_tensor_name(used_tensor_names, node_index, "rope_y1");
    const auto y2_name =
        unique_aux_tensor_name(used_tensor_names, node_index, "rope_y2");
    if (parsed.forward) {
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEY1",
          "Sub",
          {x1_cos, x2_sin},
          {y1_name},
          OrderedJson::object()));
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEY2",
          "Add",
          {x1_sin, x2_cos},
          {y2_name},
          OrderedJson::object()));
    } else {
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEY1",
          "Add",
          {x2_sin, x1_cos},
          {y1_name},
          OrderedJson::object()));
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEY2",
          "Sub",
          {x2_cos, x1_sin},
          {y2_name},
          OrderedJson::object()));
    }
    known_shapes[y1_name] = pair_shape;
    known_shapes[y2_name] = pair_shape;
    if (input_dtype.has_value()) {
      known_dtypes[y1_name] = input_dtype.value();
      known_dtypes[y2_name] = input_dtype.value();
    }

    std::string rotated_out;
    if (parsed.traditional) {
      const auto unsqueeze_axes_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "rope_pair_unsqueeze_axes",
          {4});
      const auto y1_unsqueeze = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_y1_unsqueeze");
      const auto y2_unsqueeze = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_y2_unsqueeze");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEUnsqueezeY1",
          "Unsqueeze",
          {y1_name, unsqueeze_axes_name},
          {y1_unsqueeze},
          OrderedJson::object()));
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEUnsqueezeY2",
          "Unsqueeze",
          {y2_name, unsqueeze_axes_name},
          {y2_unsqueeze},
          OrderedJson::object()));

      const auto pair_concat = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_pair_concat");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEConcatPair",
          "Concat",
          {y1_unsqueeze, y2_unsqueeze},
          {pair_concat},
          OrderedJson::object({{"axis", 4}})));

      const auto pair_shape_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "rope_pair_shape",
          {work_shape[0], work_shape[1], work_shape[2], parsed.dims});
      rotated_out =
          unique_aux_tensor_name(used_tensor_names, node_index, "rope_rotated");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEReshapePair",
          "Reshape",
          {pair_concat, pair_shape_name},
          {rotated_out},
          OrderedJson::object()));
    } else {
      rotated_out =
          unique_aux_tensor_name(used_tensor_names, node_index, "rope_rotated");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEConcatRotated",
          "Concat",
          {y1_name, y2_name},
          {rotated_out},
          OrderedJson::object({{"axis", 3}})));
    }

    known_shapes[rotated_out] = {work_shape[0], work_shape[1], work_shape[2], parsed.dims};
    if (input_dtype.has_value()) {
      known_dtypes[rotated_out] = input_dtype.value();
    }

    std::string merged_out = rotated_out;
    if (channels_known && parsed.dims < channels) {
      const auto tail_name = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_tail");
      build_slice(tail_name, parsed.dims, channels, 1);
      known_shapes[tail_name] = {
          work_shape[0], work_shape[1], work_shape[2], channels - parsed.dims};
      if (input_dtype.has_value()) {
        known_dtypes[tail_name] = input_dtype.value();
      }

      const auto concat_tail = unique_aux_tensor_name(
          used_tensor_names, node_index, "rope_with_tail");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEConcatTail",
          "Concat",
          {rotated_out, tail_name},
          {concat_tail},
          OrderedJson::object({{"axis", 3}})));
      merged_out = concat_tail;
      known_shapes[merged_out] = work_shape;
      if (input_dtype.has_value()) {
        known_dtypes[merged_out] = input_dtype.value();
      }
    } else {
      if (channels_known) {
        known_shapes[merged_out] = work_shape;
      } else {
        known_shapes[merged_out] = {
            work_shape[0], work_shape[1], work_shape[2], parsed.dims};
      }
    }

    if (input_shape.size() == 3) {
      const auto squeeze_axes_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "rope_squeeze_axes",
          {1});
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPESqueezeOutput",
          "Squeeze",
          {merged_out, squeeze_axes_name},
          outputs,
          OrderedJson::object()));
    } else {
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_RoPEOutput",
          "Identity",
          {merged_out},
          outputs,
          OrderedJson::object()));
    }

    inferred_output_shape = input_shape;
    inferred_output_dtype =
        input_dtype.has_value() ? input_dtype : known_dtype_for(known_dtypes, input_name);
    assign_known_shape_if_present(known_shapes, outputs, inferred_output_shape);
    assign_known_dtype_if_present(known_dtypes, outputs, inferred_output_dtype);
    return lowered;
  } else if (op == "Transpose") {
    const auto input_name = inputs.at(0);
    const auto input_shape = known_shape_for(known_shapes, input_name);
    if (input_shape.has_value()) {
      auto perm_opt = transpose_perm_from_arguments(arguments);
      std::vector<int64_t> perm;
      if (perm_opt.has_value() && !perm_opt->empty()) {
        perm = perm_opt.value();
      } else {
        perm.reserve(input_shape->size());
        for (int64_t i = static_cast<int64_t>(input_shape->size()) - 1; i >= 0;
             --i) {
          perm.push_back(i);
        }
      }
      inferred_output_shape =
          permute_shape(input_shape.value(), perm, "Transpose permutation");
    }
    inferred_output_dtype = known_dtype_for(known_dtypes, input_name);
  } else if (op == "Convolution") {
    return lower_onnx_convolution_node(
        node_index, inputs, outputs, arguments, lowering);
  } else if (op == "Reduce") {
    const auto reduce_code = arguments.is_array() && !arguments.empty()
        ? normalized_integer_scalar(arguments.at(0), "Reduce code")
        : 0;
    const auto axes = reduce_axes_from_arguments(arguments);
    const auto axes_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "axes", axes);

    inferred_output_shape = infer_reduce_keepdims_shape(
        known_shape_for(known_shapes, inputs.front()), axes);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs.front());

    if (reduce_code == 0 || reduce_code == 1) {
      const auto cast_bool_out =
          unique_aux_tensor_name(used_tensor_names, node_index, "cast_bool");
      const auto cast_int_out =
          unique_aux_tensor_name(used_tensor_names, node_index, "cast_int64");
      const auto reduce_out =
          unique_aux_tensor_name(used_tensor_names, node_index, "reduce");
      const auto reduce_type = onnx_reduce_op_type_from_code(reduce_code);
      if (!reduce_type.has_value()) {
        std::ostringstream out;
        out << "[ir.lowering] unsupported Reduce code " << reduce_code;
        throw std::runtime_error(out.str());
      }

      for (const auto& name : outputs) {
        if (inferred_output_shape.has_value()) {
          known_shapes[name] = inferred_output_shape.value();
        }
        known_dtypes[name] = "bool";
      }

      return {
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_CastToBool",
              "Cast",
              {inputs.front()},
              {cast_bool_out},
              OrderedJson::object({{"to", "BOOL"}})),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_CastToInt64",
              "Cast",
              {cast_bool_out},
              {cast_int_out},
              OrderedJson::object({{"to", "INT64"}})),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_" + reduce_type.value(),
              reduce_type.value(),
              {cast_int_out, axes_name},
              {reduce_out},
              OrderedJson::object({{"keepdims", 1}})),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_CastOutBool",
              "Cast",
              {reduce_out},
              outputs,
              OrderedJson::object({{"to", "BOOL"}}))};
    }

    inputs.push_back(axes_name);
    attributes["keepdims"] = 1;
  } else if (op == "AsType") {
    const auto target_dtype = onnx_effective_dtype(
        as_type_target_dtype(arguments, outputs, known_dtypes));
    attributes["to"] = onnx_dtype_symbol(target_dtype);
    inferred_output_shape = known_shape_for(known_shapes, inputs[0]);
    inferred_output_dtype = target_dtype;
  } else if (op == "Reshape") {
    const auto shape = integer_vector_argument(arguments, "Reshape");
    const auto shape_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "shape", shape);
    inputs.push_back(shape_name);
    inferred_output_shape = shape;
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  } else if (
      op == "Add" || op == "Subtract" || op == "Multiply" || op == "Divide" ||
      op == "Maximum" || op == "Minimum" || op == "Power") {
    const auto lhs_dtype = known_dtype_for(known_dtypes, inputs[0]);
    const auto rhs_dtype = known_dtype_for(known_dtypes, inputs[1]);
    const auto promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype);

    if (const auto lowered = maybe_lower_with_promoted_cast(
            node_index,
            op,
            op_type,
            inputs,
            outputs,
            attributes,
            promoted_dtype,
            std::nullopt,
            0,
            1,
            std::nullopt,
            lowering);
        lowered.has_value()) {
      return lowered.value();
    }

    inferred_output_shape = infer_elementwise_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        known_shape_for(known_shapes, inputs[1]));
    inferred_output_dtype = promoted_dtype.has_value()
        ? promoted_dtype
        : (lhs_dtype.has_value() ? lhs_dtype : rhs_dtype);
  } else if (is_passthrough_unary_op(op)) {
    infer_shape_dtype_from_input_tensor(
        inputs[0],
        known_shapes,
        known_dtypes,
        inferred_output_shape,
        inferred_output_dtype);
  } else if (op == "Sqrt") {
    infer_shape_dtype_from_input_tensor(
        inputs[0],
        known_shapes,
        known_dtypes,
        inferred_output_shape,
        inferred_output_dtype);
    if (sqrt_is_reciprocal(arguments)) {
      const auto sqrt_output =
          unique_aux_tensor_name(used_tensor_names, node_index, "sqrt");
      if (inferred_output_shape.has_value()) {
        known_shapes[sqrt_output] = inferred_output_shape.value();
      }
      if (inferred_output_dtype.has_value()) {
        known_dtypes[sqrt_output] = inferred_output_dtype.value();
      }

      return {
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_Sqrt",
              "Sqrt",
              inputs,
              {sqrt_output},
              OrderedJson::object()),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_Reciprocal",
              "Reciprocal",
              {sqrt_output},
              outputs,
              OrderedJson::object())};
    }
  } else if (op == "Matmul") {
    inferred_output_shape = infer_matmul_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        known_shape_for(known_shapes, inputs[1]));
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  } else if (op == "AddMM") {
    const auto [alpha, beta] = addmm_alpha_beta(arguments);
    if (alpha != 1.0) {
      attributes["alpha"] = alpha;
    }
    if (beta != 1.0) {
      attributes["beta"] = beta;
    }
    attributes["transA"] = 0;
    attributes["transB"] = 0;
    inferred_output_shape = infer_matmul_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        known_shape_for(known_shapes, inputs[1]));
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  } else if (op == "Square") {
    const auto square_input = inputs.at(0);
    inputs = {square_input, square_input};
    inferred_output_shape = known_shape_for(known_shapes, square_input);
    inferred_output_dtype = known_dtype_for(known_dtypes, square_input);
  } else if (op == "Gather") {
    // IR Gather shape semantics are implemented with Gather + optional
    // transpose + Unsqueeze to preserve output rank/axis behavior.
    const auto gather_state = gather_state_from_arguments(arguments, false);
    if (gather_state.has_value() &&
        (gather_state->axes.size() > 1 || inputs.size() > 2)) {
      std::vector<OrderedJson> lowered;

      for (size_t input_position = 1; input_position < inputs.size();
           ++input_position) {
        ensure_indices_input_is_int64(
            node_index,
            "GatherCastIndices" + std::to_string(input_position),
            "gather_indices_cast_" + std::to_string(input_position),
            inputs,
            input_position,
            lowered,
            lowering);
      }

      const auto& axes = gather_state->axes;
      const auto& slice_sizes = gather_state->slice_sizes;
      const Shape data_shape = require_known_static_shape_for_op(
          known_shapes, "Gather", "tensor", inputs[0]);
      const auto data_rank = static_cast<int64_t>(data_shape.size());
      if (slice_sizes.size() != data_shape.size()) {
        std::ostringstream out;
        out << "[ir.lowering] unsupported Gather slice_sizes "
            << json_from_int_vector(slice_sizes).dump() << " for tensor rank "
            << data_shape.size();
        throw std::runtime_error(out.str());
      }
      if (axes.empty()) {
        throw std::runtime_error(
            "[ir.lowering] unsupported Gather with empty axes");
      }
      if (axes.size() + 1 != inputs.size()) {
        std::ostringstream out;
        out << "[ir.lowering] unsupported Gather arity: axes=" << axes.size()
            << ", indices inputs=" << (inputs.size() - 1);
        throw std::runtime_error(out.str());
      }

      std::vector<int64_t> normalized_axes;
      normalized_axes.reserve(axes.size());
      for (const auto axis_value : axes) {
        normalized_axes.push_back(normalize_axis(
            axis_value, static_cast<size_t>(data_rank), "Gather axis"));
      }
      std::set<int64_t> unique_axes(
          normalized_axes.begin(), normalized_axes.end());
      if (unique_axes.size() != normalized_axes.size()) {
        throw std::runtime_error(
            "[ir.lowering] unsupported Gather with duplicate axes");
      }

      const Shape index_shape = require_known_static_shape_for_op(
          known_shapes, "Gather", "indices", inputs[1]);
      for (size_t input_position = 2; input_position < inputs.size();
           ++input_position) {
        const Shape next_index_shape = require_known_static_shape_for_op(
            known_shapes,
            "Gather",
            "indices",
            inputs[input_position]);
        if (next_index_shape != index_shape) {
          std::ostringstream out;
          out << "[ir.lowering] unsupported Gather with broadcasted indices: "
              << json_from_shape(index_shape).dump() << " vs "
              << json_from_shape(next_index_shape).dump();
          throw std::runtime_error(out.str());
        }
      }

      for (const auto axis_value : unique_axes) {
        if (slice_sizes[static_cast<size_t>(axis_value)] != 1) {
          std::ostringstream out;
          out << "[ir.lowering] unsupported Gather slice_sizes "
              << json_from_int_vector(slice_sizes).dump()
              << "; gathered axes must have size 1";
          throw std::runtime_error(out.str());
        }
      }
      for (size_t axis = 0; axis < data_shape.size(); ++axis) {
        if (unique_axes.find(static_cast<int64_t>(axis)) != unique_axes.end()) {
          continue;
        }
        if (slice_sizes[axis] != data_shape[axis]) {
          std::ostringstream out;
          out << "[ir.lowering] unsupported Gather slice_sizes "
              << json_from_int_vector(slice_sizes).dump()
              << "; non-gathered axis " << axis << " must match input dim "
              << data_shape[axis];
          throw std::runtime_error(out.str());
        }
      }

      std::vector<int64_t> remaining_axes;
      remaining_axes.reserve(data_shape.size() - normalized_axes.size());
      for (int64_t axis = 0; axis < data_rank; ++axis) {
        if (unique_axes.find(axis) == unique_axes.end()) {
          remaining_axes.push_back(axis);
        }
      }

      std::vector<int64_t> data_perm = normalized_axes;
      data_perm.insert(data_perm.end(), remaining_axes.begin(), remaining_axes.end());

      std::string gather_data_input = inputs[0];
      const auto input_dtype = known_dtype_for(known_dtypes, inputs[0]);
      if (!identity_permutation(data_perm)) {
        const auto transposed_data = unique_aux_tensor_name(
            used_tensor_names, node_index, "gather_data_permuted");
        lowered.push_back(build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_GatherDataTranspose",
            "Transpose",
            {inputs[0]},
            {transposed_data},
            OrderedJson::object({{"perm", json_from_int_vector(data_perm)}})));
        known_shapes[transposed_data] =
            permute_shape(data_shape, data_perm, "Gather tensor");
        if (input_dtype.has_value()) {
          known_dtypes[transposed_data] = input_dtype.value();
        }
        gather_data_input = transposed_data;
      }

      const auto index_rank = static_cast<int64_t>(index_shape.size());
      const auto index_unsqueeze_axes_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "gather_indices_unsqueeze_axes",
          {index_rank});

      std::vector<std::string> packed_indices_inputs;
      packed_indices_inputs.reserve(inputs.size() - 1);
      for (size_t input_position = 1; input_position < inputs.size();
           ++input_position) {
        const auto index_unsqueezed = unique_aux_tensor_name(
            used_tensor_names,
            node_index,
            "gather_index_unsqueezed_" + std::to_string(input_position));
        lowered.push_back(build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_GatherIndexUnsqueeze" +
                std::to_string(input_position),
            "Unsqueeze",
            {inputs[input_position], index_unsqueeze_axes_name},
            {index_unsqueezed},
            OrderedJson::object()));

        Shape unsqueezed_index_shape = index_shape;
        unsqueezed_index_shape.push_back(1);
        known_shapes[index_unsqueezed] = unsqueezed_index_shape;
        known_dtypes[index_unsqueezed] = "int64";
        packed_indices_inputs.push_back(index_unsqueezed);
      }

      const auto packed_indices = unique_aux_tensor_name(
          used_tensor_names, node_index, "gather_packed_indices");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_GatherPackIndices",
          "Concat",
          packed_indices_inputs,
          {packed_indices},
          OrderedJson::object({{"axis", index_rank}})));

      Shape packed_indices_shape = index_shape;
      packed_indices_shape.push_back(static_cast<int64_t>(packed_indices_inputs.size()));
      known_shapes[packed_indices] = packed_indices_shape;
      known_dtypes[packed_indices] = "int64";

      Shape gathernd_shape = index_shape;
      for (const auto axis_value : remaining_axes) {
        gathernd_shape.push_back(data_shape[static_cast<size_t>(axis_value)]);
      }

      const auto gathernd_output = unique_aux_tensor_name(
          used_tensor_names, node_index, "gathernd");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_GatherND",
          "GatherND",
          {gather_data_input, packed_indices},
          {gathernd_output},
          OrderedJson::object()));
      known_shapes[gathernd_output] = gathernd_shape;
      if (input_dtype.has_value()) {
        known_dtypes[gathernd_output] = input_dtype.value();
      }

      std::vector<int64_t> unsqueeze_axes;
      unsqueeze_axes.reserve(unique_axes.size());
      for (const auto axis_value : unique_axes) {
        unsqueeze_axes.push_back(index_rank + axis_value);
      }

      const auto gather_unsqueeze_axes_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "gather_result_unsqueeze_axes",
          unsqueeze_axes);
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_GatherResultUnsqueeze",
          "Unsqueeze",
          {gathernd_output, gather_unsqueeze_axes_name},
          outputs,
          OrderedJson::object()));

      Shape final_shape = index_shape;
      final_shape.insert(final_shape.end(), slice_sizes.begin(), slice_sizes.end());
      for (const auto& name : outputs) {
        known_shapes[name] = final_shape;
        if (input_dtype.has_value()) {
          known_dtypes[name] = input_dtype.value();
        }
      }

      return lowered;
    }

    const auto axis = gather_axis_from_arguments(arguments, true).value();
    std::vector<OrderedJson> pre_nodes;

    ensure_indices_input_is_int64(
        node_index,
        "GatherCastIndices",
        "gather_indices_cast",
        inputs,
        1,
        pre_nodes,
        lowering);

    const Shape data_shape = require_known_static_shape_for_op(
        known_shapes, "Gather", "tensor", inputs[0]);
    const Shape indices_shape = require_known_static_shape_for_op(
        known_shapes, "Gather", "indices", inputs[1]);

    const auto data_rank = static_cast<int64_t>(data_shape.size());
    const auto axis_index =
        normalize_axis(axis, static_cast<size_t>(data_rank), "Gather axis");
    const auto indices_rank = static_cast<int64_t>(indices_shape.size());

    const auto gather_output =
        unique_aux_tensor_name(used_tensor_names, node_index, "gather");
    const auto gather_reordered = unique_aux_tensor_name(
        used_tensor_names, node_index, "gather_reordered");
    const auto unsqueeze_axis = axis_index + indices_rank;
    const auto axes_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "axes", {unsqueeze_axis});

    const auto gather_shape = infer_gather_output_shape(
        std::optional<Shape>(data_shape),
        std::optional<Shape>(indices_shape),
        axis_index);
    const auto perm =
        gather_reorder_permutation(data_rank, indices_rank, axis_index);
    const bool needs_reorder = !identity_permutation(perm);

    std::optional<Shape> reordered_shape;
    if (gather_shape.has_value()) {
      Shape permuted;
      permuted.reserve(perm.size());
      for (const auto dim_index : perm) {
        permuted.push_back(gather_shape->at(static_cast<size_t>(dim_index)));
      }
      reordered_shape = permuted;

      known_shapes[gather_output] = gather_shape.value();
      if (needs_reorder) {
        known_shapes[gather_reordered] = reordered_shape.value();
      }

      const auto input_dtype = known_dtype_for(known_dtypes, inputs[0]);
      if (input_dtype.has_value()) {
        known_dtypes[gather_output] = input_dtype.value();
        if (needs_reorder) {
          known_dtypes[gather_reordered] = input_dtype.value();
        }
      }

      Shape final_shape = reordered_shape.value();
      final_shape.insert(
          final_shape.begin() + static_cast<long>(unsqueeze_axis), 1);
      for (const auto& name : outputs) {
        known_shapes[name] = final_shape;
        if (input_dtype.has_value()) {
          known_dtypes[name] = input_dtype.value();
        }
      }
    }

    std::vector<OrderedJson> lowered;
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_Gather",
        "Gather",
        inputs,
        {gather_output},
        attributes));

    std::string unsqueeze_input = gather_output;
    if (needs_reorder) {
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_GatherTranspose",
          "Transpose",
          {gather_output},
          {gather_reordered},
          OrderedJson::object({{"perm", json_from_int_vector(perm)}})));
      unsqueeze_input = gather_reordered;
    }

    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_Unsqueeze",
        "Unsqueeze",
        {unsqueeze_input, axes_name},
        outputs,
        OrderedJson::object()));

    pre_nodes.insert(pre_nodes.end(), lowered.begin(), lowered.end());
    return pre_nodes;
  }

  if (op == "GatherAxis") {
    // GatherAxis maps to GatherElements, with optional Expand when non-axis
    // dimensions require broadcast from singleton source dims.
    const auto axis = gather_axis_from_arguments(arguments, true).value();
    std::vector<OrderedJson> pre_nodes;

    ensure_indices_input_is_int64(
        node_index,
        "GatherAxisCastIndices",
        "gatheraxis_indices_cast",
        inputs,
        1,
        pre_nodes,
        lowering);

    const Shape data_shape = require_known_static_shape_for_op(
        known_shapes, "GatherAxis", "tensor", inputs[0]);
    const Shape indices_shape = require_known_static_shape_for_op(
        known_shapes, "GatherAxis", "indices", inputs[1]);

    const auto data_rank = static_cast<int64_t>(data_shape.size());
    const auto indices_rank = static_cast<int64_t>(indices_shape.size());
    const auto axis_index =
        normalize_axis(axis, static_cast<size_t>(data_rank), "GatherAxis axis");

    if (data_rank != indices_rank) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported GatherAxis rank mismatch: data rank "
          << data_rank << ", indices rank " << indices_rank;
      throw std::runtime_error(out.str());
    }

    const auto data_dims = data_shape;
    const auto indices_dims = indices_shape;
    Shape expanded_data_shape = data_dims;
    bool needs_data_expand = false;

    for (size_t dim_index = 0; dim_index < data_dims.size(); ++dim_index) {
      if (static_cast<int64_t>(dim_index) == axis_index) {
        continue;
      }
      const auto dim = data_dims[dim_index];
      const auto index_dim = indices_dims[dim_index];
      if (dim == index_dim) {
        continue;
      }
      if (dim == 1) {
        expanded_data_shape[dim_index] = index_dim;
        needs_data_expand = true;
        continue;
      }
      if (index_dim <= dim) {
        continue;
      }

      std::ostringstream out;
      out << "[ir.lowering] unsupported GatherAxis shape mismatch at dim "
          << dim_index << ": data=" << dim << ", indices=" << index_dim;
      throw std::runtime_error(out.str());
    }

    if (needs_data_expand) {
      const auto expand_shape_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "gatheraxis_expand_shape",
          expanded_data_shape);
      const auto expanded_data = unique_aux_tensor_name(
          used_tensor_names, node_index, "gatheraxis_expanded_data");

      known_shapes[expanded_data] = expanded_data_shape;
      const auto input_dtype = known_dtype_for(known_dtypes, inputs[0]);
      if (input_dtype.has_value()) {
        known_dtypes[expanded_data] = input_dtype.value();
      }

      inferred_output_shape = indices_dims;
      inferred_output_dtype = input_dtype;

      if (inferred_output_shape.has_value()) {
        for (const auto& name : outputs) {
          known_shapes[name] = inferred_output_shape.value();
        }
      }
      if (inferred_output_dtype.has_value()) {
        for (const auto& name : outputs) {
          known_dtypes[name] = inferred_output_dtype.value();
        }
      }

      pre_nodes.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_GatherAxisExpand",
          "Expand",
          {inputs[0], expand_shape_name},
          {expanded_data},
          OrderedJson::object()));
      pre_nodes.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_" + op_type,
          op_type,
          {expanded_data, inputs[1]},
          outputs,
          attributes));
      return pre_nodes;
    }

    inferred_output_shape = indices_dims;
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);

    if (!pre_nodes.empty()) {
      if (inferred_output_shape.has_value()) {
        for (const auto& name : outputs) {
          known_shapes[name] = inferred_output_shape.value();
        }
      }
      if (inferred_output_dtype.has_value()) {
        for (const auto& name : outputs) {
          known_dtypes[name] = inferred_output_dtype.value();
        }
      }

      pre_nodes.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_" + op_type,
          op_type,
          inputs,
          outputs,
          attributes));
      return pre_nodes;
    }
  }

  if (op == "LogSumExp") {
    const Shape input_shape = require_known_static_shape_for_op(
        known_shapes, "LogSumExp", "tensor", inputs[0]);
    const auto output_shape = known_shape_for(known_shapes, outputs.front());
    const auto axes = infer_logsumexp_axes(input_shape, output_shape);
    const auto axes_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "axes", axes);
    inputs.push_back(axes_name);
    attributes["keepdims"] = 1;
    inferred_output_shape =
        infer_reduce_keepdims_shape(std::optional<Shape>(input_shape), axes);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Pad") {
    const Shape input_shape = require_known_static_shape_for_op(
        known_shapes, "Pad", "tensor", inputs[0]);

    const auto [axes, pad_low, pad_high] =
        pad_axes_and_sizes_from_arguments(arguments, input_shape);
    const auto rank = input_shape.size();

    std::vector<int64_t> pads_begin(rank, 0);
    std::vector<int64_t> pads_end(rank, 0);
    for (size_t i = 0; i < axes.size(); ++i) {
      const auto axis_index = normalize_axis(axes[i], rank, "Pad axis");
      pads_begin[static_cast<size_t>(axis_index)] = pad_low[i];
      pads_end[static_cast<size_t>(axis_index)] = pad_high[i];
    }

    std::vector<int64_t> pads = pads_begin;
    pads.insert(pads.end(), pads_end.begin(), pads_end.end());
    const auto pads_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "pads", pads);

    if (!(inputs.size() >= 1 && inputs.size() <= 2)) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported Pad input arity " << inputs.size()
          << "; expected 1 or 2 inputs";
      throw std::runtime_error(out.str());
    }

    std::vector<std::string> padded_inputs = {inputs.front(), pads_name};
    if (inputs.size() == 2) {
      padded_inputs.push_back(inputs[1]);
    }
    inputs = std::move(padded_inputs);

    attributes["mode"] = "constant";
    inferred_output_shape = infer_pad_output_shape(
        std::optional<Shape>(input_shape), pads_begin, pads_end);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs.front());
  }

  if (op == "Scan") {
    const auto parsed = scan_arguments(arguments);
    if (parsed.reduce_type != 2) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported Scan reduce_type "
          << parsed.reduce_type << "; only CumSum (2) is supported";
      throw std::runtime_error(out.str());
    }

    int64_t axis_value = parsed.axis;
    const auto input_shape = known_shape_for(known_shapes, inputs[0]);
    if (input_shape.has_value()) {
      axis_value =
          normalize_axis(parsed.axis, input_shape->size(), "Scan axis");
    }

    const auto axis_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "axis", {axis_value});
    inputs.push_back(axis_name);
    if (!parsed.inclusive) {
      attributes["exclusive"] = 1;
    }
    if (parsed.reverse) {
      attributes["reverse"] = 1;
    }

    inferred_output_shape = known_shape_for(known_shapes, inputs[0]);
    auto output_dtype = known_dtype_for(known_dtypes, outputs.front());
    inferred_output_dtype = output_dtype.has_value()
        ? output_dtype
        : known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Slice") {
    const auto [starts, ends, axes, steps] =
        slice_vectors_from_arguments(arguments);
    const auto starts_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "starts", starts);
    const auto ends_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "ends", ends);
    const auto axes_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "axes", axes);
    const auto steps_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "steps", steps);

    inputs.push_back(starts_name);
    inputs.push_back(ends_name);
    inputs.push_back(axes_name);
    inputs.push_back(steps_name);

    inferred_output_shape = infer_slice_output_shape(
        known_shape_for(known_shapes, inputs[0]), starts, ends, axes, steps);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "SliceUpdate") {
    if (inputs.size() != 2) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported SliceUpdate input arity "
          << inputs.size() << "; expected 2";
      throw std::runtime_error(out.str());
    }

    const auto [starts, ends, axes, steps] =
        slice_vectors_from_arguments(arguments);
    const Shape source_shape = require_known_static_shape_for_op(
        known_shapes, "SliceUpdate", "tensor", inputs[0]);
    if (starts.size() != source_shape.size()) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported SliceUpdate arguments "
          << arguments.dump() << " for rank " << source_shape.size();
      throw std::runtime_error(out.str());
    }
    if (std::any_of(steps.begin(), steps.end(), [](int64_t step) {
          return step <= 0;
        })) {
      throw std::runtime_error(
          "[ir.lowering] unsupported SliceUpdate with non-positive stride");
    }

    const auto update_shape = infer_slice_output_shape(
        std::optional<Shape>(source_shape), starts, ends, axes, steps);
    if (!update_shape.has_value()) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported SliceUpdate arguments "
          << arguments.dump();
      throw std::runtime_error(out.str());
    }

    const auto source_rank = static_cast<int64_t>(source_shape.size());
    std::vector<int64_t> normalized_starts(source_shape.size(), 0);
    std::vector<int64_t> normalized_steps(source_shape.size(), 1);
    for (size_t i = 0; i < axes.size(); ++i) {
      const auto axis_index =
          normalize_axis(axes[i], source_shape.size(), "SliceUpdate axis");
      normalized_starts[static_cast<size_t>(axis_index)] =
          normalize_slice_index(starts[i], source_shape[static_cast<size_t>(axis_index)]);
      normalized_steps[static_cast<size_t>(axis_index)] = steps[i];
    }

    const int64_t update_size = tensor_size_from_shape(update_shape.value());
    std::vector<int64_t> scatter_indices;
    scatter_indices.reserve(
        static_cast<size_t>(update_size * source_rank));

    std::vector<int64_t> logical_index(update_shape->size(), 0);
    for (int64_t linear = 0; linear < update_size; ++linear) {
      int64_t remainder = linear;
      for (int64_t axis = static_cast<int64_t>(update_shape->size()) - 1;
           axis >= 0;
           --axis) {
        const auto axis_index = static_cast<size_t>(axis);
        const auto dim = update_shape->at(axis_index);
        if (dim == 0) {
          logical_index[axis_index] = 0;
          continue;
        }
        logical_index[axis_index] = remainder % dim;
        remainder /= dim;
      }

      for (size_t axis = 0; axis < source_shape.size(); ++axis) {
        const auto coord =
            normalized_starts[axis] +
            logical_index[axis] * normalized_steps[axis];
        if (coord < 0 || coord >= source_shape[axis]) {
          std::ostringstream out;
          out << "[ir.lowering] unsupported SliceUpdate coordinate " << coord
              << " at axis " << axis << " for dim " << source_shape[axis];
          throw std::runtime_error(out.str());
        }
        scatter_indices.push_back(coord);
      }
    }

    std::vector<OrderedJson> lowered;
    auto update_input = inputs[1];
    const auto source_dtype = canonical_dtype(known_dtype_for(known_dtypes, inputs[0]));
    const auto update_dtype = canonical_dtype(known_dtype_for(known_dtypes, update_input));
    if (source_dtype.has_value() && update_dtype.has_value() &&
        source_dtype.value() != update_dtype.value()) {
      const auto cast_update = unique_aux_tensor_name(
          used_tensor_names, node_index, "sliceupdate_update_cast");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_SliceUpdateCastUpdate",
          "Cast",
          {update_input},
          {cast_update},
          OrderedJson::object(
              {{"to", onnx_dtype_symbol(source_dtype.value())}})));
      known_shapes[cast_update] = update_shape.value();
      known_dtypes[cast_update] = source_dtype.value();
      update_input = cast_update;
    }

    const auto updates_flat_shape = append_aux_int64_initializer(
        initializers,
        used_tensor_names,
        node_index,
        "sliceupdate_updates_flat_shape",
        {-1});
    const auto updates_flat = unique_aux_tensor_name(
        used_tensor_names, node_index, "sliceupdate_updates_flat");
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_SliceUpdateFlattenUpdates",
        "Reshape",
        {update_input, updates_flat_shape},
        {updates_flat},
        OrderedJson::object()));
    known_shapes[updates_flat] = {update_size};
    if (source_dtype.has_value()) {
      known_dtypes[updates_flat] = source_dtype.value();
    } else if (update_dtype.has_value()) {
      known_dtypes[updates_flat] = update_dtype.value();
    }

    const auto indices_name = unique_aux_tensor_name(
        used_tensor_names, node_index, "sliceupdate_indices");
    OrderedJson indices_tensor = OrderedJson::object();
    indices_tensor["name"] = indices_name;
    indices_tensor["shape"] =
        json_from_int_vector({update_size, source_rank});
    indices_tensor["dtype"] = "int64";
    indices_tensor["values"] = json_from_int_vector(scatter_indices);
    initializers.push_back(onnx_initializer_info(indices_tensor));

    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_SliceUpdateScatter",
        "ScatterND",
        {inputs[0], indices_name, updates_flat},
        outputs,
        OrderedJson::object()));

    for (const auto& name : outputs) {
      known_shapes[name] = source_shape;
      if (source_dtype.has_value()) {
        known_dtypes[name] = source_dtype.value();
      }
    }

    return lowered;
  }

  if (op == "Split") {
    const auto [axis, lengths] = split_axis_and_lengths(
        arguments,
        known_shape_for(known_shapes, inputs[0]),
        static_cast<int64_t>(outputs.size()));
    const auto split_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "split", lengths);
    inputs.push_back(split_name);
    attributes["axis"] = axis;

    const auto split_shapes = infer_split_output_shapes(
        known_shape_for(known_shapes, inputs[0]), axis, lengths);
    if (split_shapes.has_value()) {
      for (size_t i = 0; i < outputs.size(); ++i) {
        known_shapes[outputs[i]] = split_shapes->at(i);
      }
    }
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "ArgReduce") {
    // IR ArgReduce expects uint32 output; ONNX ArgMin/ArgMax return int64.
    const auto [arg_mode, arg_axis] = argreduce_mode_axis(arguments);
    const auto arg_op = onnx_argreduce_op_type_from_code(arg_mode);
    if (!arg_op.has_value()) {
      std::ostringstream out;
      out << "[ir.lowering] unsupported ArgReduce code " << arg_mode;
      throw std::runtime_error(out.str());
    }
    const auto arg_output =
        unique_aux_tensor_name(used_tensor_names, node_index, "argreduce");

    const auto arg_shape = infer_argreduce_keepdims_shape(
        known_shape_for(known_shapes, inputs[0]), arg_axis);
    if (arg_shape.has_value()) {
      known_shapes[arg_output] = arg_shape.value();
      for (const auto& name : outputs) {
        known_shapes[name] = arg_shape.value();
      }
    }

    return {
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_" + arg_op.value(),
            arg_op.value(),
            inputs,
            {arg_output},
            OrderedJson::object({{"axis", arg_axis}, {"keepdims", 1}})),
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_CastUint32",
            "Cast",
            {arg_output},
            outputs,
            OrderedJson::object({{"to", "UINT32"}}))};
  }

  if (op == "AsStrided") {
    // AsStrided lowering computes linear gather indices in host code and emits
    // Gather over flattened input.
    const auto input_name = inputs.at(0);
    const Shape input_shape = require_known_static_shape_for_op(
        known_shapes, "AsStrided", "tensor", input_name);

    const auto parsed = asstrided_arguments(arguments);
    const auto input_size = tensor_size_from_shape(input_shape);
    const auto indices = asstrided_linear_indices(
        parsed.output_shape, parsed.strides, parsed.offset, input_size);

    const auto indices_name = unique_aux_tensor_name(
        used_tensor_names, node_index, "asstrided_indices");
    OrderedJson indices_tensor = OrderedJson::object();
    indices_tensor["name"] = indices_name;
    indices_tensor["shape"] = json_from_shape(parsed.output_shape);
    indices_tensor["dtype"] = "int64";
    indices_tensor["values"] = json_from_int_vector(indices);
    initializers.push_back(onnx_initializer_info(indices_tensor));

    const auto input_rank = static_cast<int64_t>(input_shape.size());
    auto gather_input = input_name;
    std::vector<OrderedJson> lowered;
    if (input_rank != 1) {
      const auto flatten_shape_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "asstrided_flatten_shape",
          {-1});
      gather_input = unique_aux_tensor_name(
          used_tensor_names, node_index, "asstrided_input_flat");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_AsStridedInputFlatten",
          "Reshape",
          {input_name, flatten_shape_name},
          {gather_input},
          OrderedJson::object()));
    }

    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_AsStridedGather",
        "Gather",
        {gather_input, indices_name},
        outputs,
        OrderedJson::object({{"axis", 0}})));

    const auto input_dtype = known_dtype_for(known_dtypes, input_name);
    for (const auto& name : outputs) {
      known_shapes[name] = parsed.output_shape;
      if (input_dtype.has_value()) {
        known_dtypes[name] = input_dtype.value();
      }
    }

    return lowered;
  }

  if (op == "ScatterAxis") {
    infer_shape_dtype_from_input_tensor(
        inputs[0],
        known_shapes,
        known_dtypes,
        inferred_output_shape,
        inferred_output_dtype);
  }

  if (op == "Greater" || op == "GreaterEqual" || op == "Less") {
    const auto lhs_dtype = known_dtype_for(known_dtypes, inputs[0]);
    const auto rhs_dtype = known_dtype_for(known_dtypes, inputs[1]);
    const auto promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype);
    if (const auto lowered = maybe_lower_with_promoted_cast(
            node_index,
            op,
            op_type,
            inputs,
            outputs,
            attributes,
            promoted_dtype,
            std::nullopt,
            0,
            1,
            std::optional<std::string>("bool"),
            lowering);
        lowered.has_value()) {
      return lowered.value();
    }

    inferred_output_shape = infer_elementwise_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        known_shape_for(known_shapes, inputs[1]));
    inferred_output_dtype = "bool";
  }

  if (op == "Equal") {
    if (equal_nan_from_arguments(arguments)) {
      throw std::runtime_error(
          "[ir.lowering] unsupported Equal equal_nan=true; only equal_nan=false is supported");
    }

    const auto lhs_dtype = known_dtype_for(known_dtypes, inputs[0]);
    const auto rhs_dtype = known_dtype_for(known_dtypes, inputs[1]);
    const auto promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype);

    if (const auto lowered = maybe_lower_with_promoted_cast(
            node_index,
            op,
            op_type,
            inputs,
            outputs,
            attributes,
            promoted_dtype,
            std::nullopt,
            0,
            1,
            std::optional<std::string>("bool"),
            lowering);
        lowered.has_value()) {
      return lowered.value();
    }

    inferred_output_shape = infer_elementwise_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        known_shape_for(known_shapes, inputs[1]));
    inferred_output_dtype = "bool";
  }

  if (op == "Select") {
    const auto lhs_dtype = known_dtype_for(known_dtypes, inputs[1]);
    const auto rhs_dtype = known_dtype_for(known_dtypes, inputs[2]);
    const auto promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype);

    if (const auto lowered = maybe_lower_with_promoted_cast(
            node_index,
            op,
            op_type,
            inputs,
            outputs,
            attributes,
            promoted_dtype,
            std::optional<std::vector<size_t>>{{1, 2}},
            1,
            2,
            std::nullopt,
            lowering);
        lowered.has_value()) {
      return lowered.value();
    }

    inferred_output_shape = infer_elementwise_output_shape(
        known_shape_for(known_shapes, inputs[1]),
        known_shape_for(known_shapes, inputs[2]));
    inferred_output_dtype = promoted_dtype.has_value()
        ? promoted_dtype
        : (lhs_dtype.has_value() ? lhs_dtype : rhs_dtype);
  }

  if (op == "Full") {
    infer_shape_dtype_from_input_tensor(
        inputs[0],
        known_shapes,
        known_dtypes,
        inferred_output_shape,
        inferred_output_dtype);
  }

  if (op == "Concatenate") {
    const auto axis = concatenate_axis_from_arguments(arguments, true).value();
    std::vector<std::optional<Shape>> input_shapes;
    input_shapes.reserve(inputs.size());
    for (const auto& name : inputs) {
      input_shapes.push_back(known_shape_for(known_shapes, name));
    }
    inferred_output_shape = infer_concatenate_output_shape(input_shapes, axis);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs.front());
  }

  if (op == "Flatten") {
    const auto flatten_input = inputs.front();
    const Shape input_shape = require_known_static_shape_for_op(
        known_shapes, "Flatten", "tensor", flatten_input);
    const auto shape = flatten_shape_from_arguments(arguments, input_shape);
    const auto shape_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "shape", shape);
    inputs.push_back(shape_name);
    inferred_output_shape = shape;
    inferred_output_dtype = known_dtype_for(known_dtypes, flatten_input);
  }

  if (op == "Unflatten") {
    const auto unflatten_input = inputs.front();
    const auto input_shape = known_shape_for(known_shapes, unflatten_input);
    if (input_shape.has_value()) {
      const auto shape =
          unflatten_shape_from_arguments(arguments, input_shape.value());
      const auto shape_name = append_aux_int64_initializer(
          initializers, used_tensor_names, node_index, "shape", shape);
      inputs.push_back(shape_name);
      inferred_output_shape = shape;
      inferred_output_dtype = known_dtype_for(known_dtypes, unflatten_input);
    } else {
      auto [axis, target_shape] = unflatten_axis_shape_from_arguments(arguments);
      if (axis < 0) {
        std::ostringstream out;
        out << "[ir.lowering] unsupported Unflatten axis " << axis
            << " for tensor " << unflatten_input
            << " without known static shape";
        throw std::runtime_error(out.str());
      }

      std::vector<OrderedJson> lowered;
      const auto input_dtype = known_dtype_for(known_dtypes, unflatten_input);

      const auto input_shape_name = unique_aux_tensor_name(
          used_tensor_names, node_index, "unflatten_input_shape");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_UnflattenShape",
          "Shape",
          {unflatten_input},
          {input_shape_name},
          OrderedJson::object()));
      known_dtypes[input_shape_name] = "int64";

      const auto prefix_starts_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "unflatten_prefix_starts",
          {0});
      const auto prefix_ends_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "unflatten_prefix_ends",
          {axis});
      const auto prefix_axes_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "unflatten_prefix_axes",
          {0});
      const auto prefix_steps_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "unflatten_prefix_steps",
          {1});
      const auto prefix_shape_name = unique_aux_tensor_name(
          used_tensor_names, node_index, "unflatten_prefix_shape");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_UnflattenPrefix",
          "Slice",
          {input_shape_name,
           prefix_starts_name,
           prefix_ends_name,
           prefix_axes_name,
           prefix_steps_name},
          {prefix_shape_name},
          OrderedJson::object()));
      known_dtypes[prefix_shape_name] = "int64";

      const auto suffix_starts_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "unflatten_suffix_starts",
          {axis + 1});
      const auto suffix_ends_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "unflatten_suffix_ends",
          {std::numeric_limits<int64_t>::max()});
      const auto suffix_axes_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "unflatten_suffix_axes",
          {0});
      const auto suffix_steps_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "unflatten_suffix_steps",
          {1});
      const auto suffix_shape_name = unique_aux_tensor_name(
          used_tensor_names, node_index, "unflatten_suffix_shape");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_UnflattenSuffix",
          "Slice",
          {input_shape_name,
           suffix_starts_name,
           suffix_ends_name,
           suffix_axes_name,
           suffix_steps_name},
          {suffix_shape_name},
          OrderedJson::object()));
      known_dtypes[suffix_shape_name] = "int64";

      const auto target_shape_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "unflatten_target_shape",
          target_shape);
      const auto reshape_shape_name = unique_aux_tensor_name(
          used_tensor_names, node_index, "unflatten_shape");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_UnflattenConcatShape",
          "Concat",
          {prefix_shape_name, target_shape_name, suffix_shape_name},
          {reshape_shape_name},
          OrderedJson::object({{"axis", 0}})));
      known_dtypes[reshape_shape_name] = "int64";

      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_UnflattenReshape",
          "Reshape",
          {unflatten_input, reshape_shape_name},
          outputs,
          OrderedJson::object()));

      if (input_dtype.has_value()) {
        for (const auto& name : outputs) {
          known_dtypes[name] = input_dtype.value();
        }
      }

      return lowered;
    }
  }

  if (op == "Squeeze") {
    const auto axes = integer_vector_argument(arguments, "Squeeze");
    const auto axes_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "axes", axes);
    inputs.push_back(axes_name);
    inferred_output_shape = infer_squeeze_output_shape(
        known_shape_for(known_shapes, inputs[0]), axes);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "ExpandDims") {
    const auto axes = integer_vector_argument(arguments, "ExpandDims");
    const auto axes_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "axes", axes);
    inputs.push_back(axes_name);
    inferred_output_shape = infer_unsqueeze_output_shape(
        known_shape_for(known_shapes, inputs[0]), axes);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Broadcast") {
    // BF16 Expand is normalized via fp32 Cast -> Expand -> Cast for backend
    // compatibility.
    const auto shape = integer_vector_argument(arguments, "Broadcast");
    const auto shape_name = append_aux_int64_initializer(
        initializers, used_tensor_names, node_index, "shape", shape);
    const auto input_name = inputs.at(0);
    const auto input_dtype = known_dtype_for(known_dtypes, input_name);

    if (input_dtype.has_value() && input_dtype.value() == "bfloat16") {
      const auto cast_input = unique_aux_tensor_name(
          used_tensor_names, node_index, "broadcast_cast_input");
      const auto expand_output = unique_aux_tensor_name(
          used_tensor_names, node_index, "broadcast_expand_output");

      const auto input_shape = known_shape_for(known_shapes, input_name);
      if (input_shape.has_value()) {
        known_shapes[cast_input] = input_shape.value();
      }
      known_shapes[expand_output] = shape;
      known_dtypes[cast_input] = "float32";
      known_dtypes[expand_output] = "float32";

      for (const auto& name : outputs) {
        known_shapes[name] = shape;
        known_dtypes[name] = "bfloat16";
      }

      return {
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_BroadcastCastInput",
              "Cast",
              {input_name},
              {cast_input},
              OrderedJson::object({{"to", onnx_dtype_symbol("float32")}})),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_Expand",
              "Expand",
              {cast_input, shape_name},
              {expand_output},
              OrderedJson::object()),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_BroadcastCastOutput",
              "Cast",
              {expand_output},
              outputs,
              OrderedJson::object({{"to", onnx_dtype_symbol("bfloat16")}}))};
    }

    inputs.push_back(shape_name);
    inferred_output_shape = shape;
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  assign_known_shape_if_present(known_shapes, outputs, inferred_output_shape);
  assign_known_dtype_if_present(known_dtypes, outputs, inferred_output_dtype);

  return {build_onnx_node_spec(
      "node_" + std::to_string(node_index) + "_" + op_type,
      op_type,
      inputs,
      outputs,
      attributes)};
}

} // namespace detail
} // namespace mlx::onnx



namespace mlx::onnx::detail {

static OrderedJson ir_to_onnx_json_payload_impl(
    const OrderedJson& payload,
    int64_t opset,
    const std::string& model_name) {
  // Graph-wide lowering orchestrator. Assumes payload was already validated by
  // capture/parser layers and emits a deterministic ONNX stub JSON envelope.
  OrderedJson initializers = OrderedJson::array();
  for (const auto& tensor : payload.at("constants")) {
    initializers.push_back(onnx_initializer_info(tensor));
  }

  auto used_tensor_names = collect_payload_tensor_names(payload);
  auto known_shapes = collect_known_tensor_shapes(payload);
  auto known_dtypes = collect_known_tensor_dtypes(payload);
  LoweringContext lowering{
      initializers, used_tensor_names, known_shapes, known_dtypes};

  OrderedJson nodes = OrderedJson::array();
  const auto& source_nodes = payload.at("nodes");
  for (size_t index = 0; index < source_nodes.size(); ++index) {
    auto lowered =
        lower_onnx_node_default(source_nodes.at(index), index, lowering);
    for (auto& lowered_node : lowered) {
      nodes.push_back(std::move(lowered_node));
    }
  }

  OrderedJson graph = OrderedJson::object();
  graph["name"] = model_name;

  OrderedJson input_infos = OrderedJson::array();
  for (const auto& tensor : payload.at("inputs")) {
    input_infos.push_back(onnx_value_info(tensor));
  }
  graph["inputs"] = std::move(input_infos);

  OrderedJson output_infos = OrderedJson::array();
  for (const auto& tensor : payload.at("outputs")) {
    output_infos.push_back(onnx_value_info(tensor));
  }
  graph["outputs"] = std::move(output_infos);

  graph["initializers"] = std::move(initializers);
  graph["nodes"] = std::move(nodes);

  OrderedJson out = OrderedJson::object();
  out["format"] = "onnx_stub_v1";
  out["ir_version"] = kGraphIrVersion;
  out["opset"] = opset;
  out["producer_name"] = "mlx-ruby";
  out["graph"] = std::move(graph);
  return out;
}

} // namespace mlx::onnx::detail

namespace mlx::onnx {

OrderedJson ir_to_onnx_json_payload(
    const OrderedJson& payload,
    int64_t opset,
    const std::string& model_name) {
  return detail::ir_to_onnx_json_payload_impl(payload, opset, model_name);
}

OrderedJson ir_compatibility_report_payload(const OrderedJson& payload) {
  return detail::ir_compatibility_report_payload_impl(payload);
}

OnnxBinaryArtifact build_onnx_binary_artifact_from_stub(
    const OrderedJson& onnx_stub,
    const OnnxBinaryWriteOptions& options) {
  return detail::build_onnx_binary_artifact_from_stub_impl(onnx_stub, options);
}

} // namespace mlx::onnx

#include "detail.hpp"
#include "mappings.hpp"

#include <array>
#include <bit>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mlx::onnx::detail {

struct OnnxAttributeModel {
  std::string name;
  OrderedJson value;
};

struct OnnxNodeModel {
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<OnnxAttributeModel> attributes;
};

struct OnnxValueInfoModel {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype;
  int elem_type = 0;
};

struct OnnxInitializerModel {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype;
  int elem_type = 0;
  OrderedJson values;
};

struct OnnxGraphModel {
  std::string name;
  std::vector<OnnxNodeModel> nodes;
  std::vector<OnnxInitializerModel> initializers;
  std::vector<OnnxValueInfoModel> inputs;
  std::vector<OnnxValueInfoModel> outputs;
};

struct OnnxStubModel {
  int64_t opset = 0;
  std::string producer_name;
  OnnxGraphModel graph;
};

enum class PbWireType : uint8_t {
  kVarint = 0,
  kFixed64 = 1,
  kLengthDelimited = 2,
  kFixed32 = 5,
};

OnnxStubModel onnx_stub_model_from_json(const OrderedJson& onnx_stub);

void pb_write_varint(std::string& out, uint64_t value);
void pb_write_key(std::string& out, int field_number, PbWireType wire_type);
void pb_write_varint_field(std::string& out, int field_number, uint64_t value);
void pb_write_int64_field(std::string& out, int field_number, int64_t value);
void pb_write_string_field(
    std::string& out,
    int field_number,
    const std::string& value);
void pb_write_bytes_field(
    std::string& out,
    int field_number,
    const std::string& value);
void pb_write_message_field(
    std::string& out,
    int field_number,
    const std::string& message);
void pb_write_fixed32_field(std::string& out, int field_number, uint32_t value);

std::string pb_encode_tensor(
    const OnnxInitializerModel& tensor,
    const OnnxBinaryWriteOptions& options,
    std::string& external_data,
    uint64_t& external_offset,
    bool& has_external_data);

} // namespace mlx::onnx::detail




namespace mlx::onnx::detail {
namespace {

std::vector<int64_t> parse_shape(
    const OrderedJson& shape,
    const std::string& label) {
  if (!shape.is_array()) {
    throw std::invalid_argument(label + " shape must be an Array");
  }
  return normalize_integer_vector(shape, label + " shape dim");
}

int parse_elem_type_for_dtype(
    const std::string& dtype,
    const std::string& label) {
  const auto symbol = onnx_dtype_symbol(onnx_effective_dtype(dtype));
  try {
    return onnx_elem_type_from_symbol_lookup(symbol);
  } catch (const std::exception& error) {
    throw std::invalid_argument(
        label + " unsupported dtype " + dtype + ": " + error.what());
  }
}

int parse_elem_type_from_value_info(const OrderedJson& info) {
  if (info.contains("onnx_elem_type") &&
      info.at("onnx_elem_type").is_string()) {
    return onnx_elem_type_from_symbol_lookup(
        info.at("onnx_elem_type").get<std::string>());
  }
  return parse_elem_type_for_dtype(
      info.at("dtype").get<std::string>(), "value_info");
}

OnnxValueInfoModel parse_value_info(const OrderedJson& info) {
  if (!info.is_object()) {
    throw std::invalid_argument("value_info must be an Object");
  }
  OnnxValueInfoModel out;
  out.name = info.at("name").get<std::string>();
  out.shape = parse_shape(info.at("shape"), "value_info");
  out.dtype = info.at("dtype").get<std::string>();
  out.elem_type = parse_elem_type_from_value_info(info);
  return out;
}

OnnxAttributeModel parse_attribute(
    const std::string& name,
    const OrderedJson& value) {
  OnnxAttributeModel out;
  out.name = name;
  out.value = value;
  return out;
}

OnnxNodeModel parse_node(const OrderedJson& node) {
  if (!node.is_object()) {
    throw std::invalid_argument("node must be an Object");
  }
  OnnxNodeModel out;
  out.name = node.at("name").get<std::string>();
  out.op_type = node.at("op_type").get<std::string>();
  out.inputs = parse_string_array(node.at("inputs"), "node inputs");
  out.outputs = parse_string_array(node.at("outputs"), "node outputs");
  const auto& attrs = node.at("attributes");
  if (!attrs.is_object()) {
    throw std::invalid_argument("node attributes must be an Object");
  }
  for (auto it = attrs.begin(); it != attrs.end(); ++it) {
    out.attributes.push_back(parse_attribute(it.key(), it.value()));
  }
  return out;
}

OnnxInitializerModel parse_initializer(const OrderedJson& initializer) {
  if (!initializer.is_object()) {
    throw std::invalid_argument("initializer must be an Object");
  }
  OnnxInitializerModel out;
  out.name = initializer.at("name").get<std::string>();
  out.shape = parse_shape(initializer.at("shape"), "initializer");
  out.dtype = initializer.at("dtype").get<std::string>();
  out.elem_type = parse_elem_type_for_dtype(out.dtype, "initializer");
  out.values = initializer.at("values");
  return out;
}

OnnxGraphModel parse_graph(const OrderedJson& graph) {
  if (!graph.is_object()) {
    throw std::invalid_argument("onnx stub graph must be an Object");
  }
  OnnxGraphModel out;
  out.name = graph.at("name").get<std::string>();

  const auto& nodes = graph.at("nodes");
  if (!nodes.is_array()) {
    throw std::invalid_argument("onnx stub graph nodes must be an Array");
  }
  out.nodes.reserve(nodes.size());
  for (const auto& node : nodes) {
    out.nodes.push_back(parse_node(node));
  }

  const auto& initializers = graph.at("initializers");
  if (!initializers.is_array()) {
    throw std::invalid_argument(
        "onnx stub graph initializers must be an Array");
  }
  out.initializers.reserve(initializers.size());
  for (const auto& initializer : initializers) {
    out.initializers.push_back(parse_initializer(initializer));
  }

  const auto& inputs = graph.at("inputs");
  if (!inputs.is_array()) {
    throw std::invalid_argument("onnx stub graph inputs must be an Array");
  }
  out.inputs.reserve(inputs.size());
  for (const auto& input : inputs) {
    out.inputs.push_back(parse_value_info(input));
  }

  const auto& outputs = graph.at("outputs");
  if (!outputs.is_array()) {
    throw std::invalid_argument("onnx stub graph outputs must be an Array");
  }
  out.outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    out.outputs.push_back(parse_value_info(output));
  }

  return out;
}

} // namespace

OnnxStubModel onnx_stub_model_from_json(const OrderedJson& onnx_stub) {
  if (!onnx_stub.is_object()) {
    throw std::invalid_argument("onnx stub must be a JSON object");
  }
  if (!onnx_stub.contains("graph") || !onnx_stub.at("graph").is_object()) {
    throw std::invalid_argument("onnx stub must include graph object");
  }

  OnnxStubModel out;
  out.opset =
      normalized_integer_scalar(onnx_stub.at("opset"), "onnx_stub opset");
  out.producer_name = onnx_stub.contains("producer_name") &&
          onnx_stub.at("producer_name").is_string()
      ? onnx_stub.at("producer_name").get<std::string>()
      : "mlx-ruby";
  out.graph = parse_graph(onnx_stub.at("graph"));
  return out;
}

} // namespace mlx::onnx::detail



namespace mlx::onnx::detail {

void pb_write_varint(std::string& out, uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7fU) | 0x80U));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}

void pb_write_key(std::string& out, int field_number, PbWireType wire_type) {
  const uint64_t key = (static_cast<uint64_t>(field_number) << 3) |
      static_cast<uint64_t>(wire_type);
  pb_write_varint(out, key);
}

void pb_write_varint_field(std::string& out, int field_number, uint64_t value) {
  pb_write_key(out, field_number, PbWireType::kVarint);
  pb_write_varint(out, value);
}

void pb_write_int64_field(std::string& out, int field_number, int64_t value) {
  pb_write_varint_field(out, field_number, static_cast<uint64_t>(value));
}

void pb_write_string_field(
    std::string& out,
    int field_number,
    const std::string& value) {
  pb_write_key(out, field_number, PbWireType::kLengthDelimited);
  pb_write_varint(out, static_cast<uint64_t>(value.size()));
  out.append(value);
}

void pb_write_bytes_field(
    std::string& out,
    int field_number,
    const std::string& value) {
  pb_write_string_field(out, field_number, value);
}

void pb_write_message_field(
    std::string& out,
    int field_number,
    const std::string& message) {
  pb_write_key(out, field_number, PbWireType::kLengthDelimited);
  pb_write_varint(out, static_cast<uint64_t>(message.size()));
  out.append(message);
}

void pb_write_fixed32_field(
    std::string& out,
    int field_number,
    uint32_t value) {
  pb_write_key(out, field_number, PbWireType::kFixed32);
  std::array<char, 4> bytes = {
      static_cast<char>(value & 0xffU),
      static_cast<char>((value >> 8) & 0xffU),
      static_cast<char>((value >> 16) & 0xffU),
      static_cast<char>((value >> 24) & 0xffU)};
  out.append(bytes.data(), bytes.size());
}

} // namespace mlx::onnx::detail



namespace mlx::onnx::detail {
namespace {

bool json_integer_like(const OrderedJson& value) {
  if (value.is_number_integer() || value.is_number_unsigned()) {
    return true;
  }
  if (value.is_number_float()) {
    const double v = value.get<double>();
    return std::isfinite(v) && std::trunc(v) == v;
  }
  return false;
}

size_t expected_initializer_value_count(const std::vector<int64_t>& dims) {
  if (dims.empty()) {
    return 1;
  }
  size_t total = 1;
  for (const auto dim : dims) {
    if (dim < 0) {
      throw std::invalid_argument(
          "initializer shape values must be non-negative");
    }
    total *= static_cast<size_t>(dim);
  }
  return total;
}

void collect_initializer_leaves(
    const OrderedJson& value,
    std::vector<const OrderedJson*>& out) {
  if (value.is_array()) {
    for (const auto& item : value) {
      collect_initializer_leaves(item, out);
    }
    return;
  }
  out.push_back(&value);
}

uint16_t float32_to_float16_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t sign = (bits >> 16) & 0x8000U;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffU;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t half_mantissa = mantissa >> shift;
    if ((mantissa >> (shift - 1)) & 1U) {
      half_mantissa += 1U;
    }
    return static_cast<uint16_t>(sign | half_mantissa);
  }

  if (exponent >= 0x1f) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }

  uint16_t half = static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
  if (mantissa & 0x00001000U) {
    half = static_cast<uint16_t>(half + 1U);
  }
  return half;
}

uint16_t float32_to_bfloat16_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x00008000U) >> 16);
}

template <typename T>
void append_le_bytes(std::string& out, T value) {
  std::array<char, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  out.append(bytes.data(), bytes.size());
}

void raise_invalid_complex_literal(const std::string& label) {
  std::ostringstream out;
  out << label << " unsupported complex literal";
  throw std::invalid_argument(out.str());
}

std::string normalize_complex_literal(std::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (const char ch : value) {
    if (!std::isspace(static_cast<unsigned char>(ch))) {
      normalized.push_back(ch);
    }
  }
  return normalized;
}

std::size_t find_complex_literal_split(std::string_view value) {
  for (std::size_t idx = value.size(); idx > 0; --idx) {
    const std::size_t pos = idx - 1;
    if (pos == 0) {
      continue;
    }
    const char ch = value[pos];
    if ((ch == '+' || ch == '-') && value[pos - 1] != 'e' &&
        value[pos - 1] != 'E') {
      return pos;
    }
  }
  return std::string_view::npos;
}

double parse_complex_literal_double(
    const std::string& text,
    const std::string& label) {
  if (text.empty()) {
    raise_invalid_complex_literal(label);
  }
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(text.c_str(), &end);
  if (text.c_str() == end) {
    raise_invalid_complex_literal(label);
  }
  if (errno == ERANGE) {
    raise_invalid_complex_literal(label);
  }
  if (static_cast<size_t>(end - text.c_str()) != text.size()) {
    raise_invalid_complex_literal(label);
  }
  return value;
}

std::pair<float, float> complex64_pair_from_string(
    const std::string& raw,
    const std::string& label) {
  const std::string normalized = normalize_complex_literal(raw);
  if (normalized.empty()) {
    raise_invalid_complex_literal(label);
  }
  const char last = normalized.back();
  if (last != 'i' && last != 'I') {
    raise_invalid_complex_literal(label);
  }
  std::string_view remaining(normalized.data(), normalized.size() - 1);
  if (remaining.empty()) {
    return {0.0f, 1.0f};
  }
  const std::size_t split = find_complex_literal_split(remaining);
  std::string real_text;
  std::string imag_text;
  if (split == std::string_view::npos) {
    real_text = "0";
    if (remaining == "+" || remaining == "-") {
      imag_text = remaining == "+" ? "1" : "-1";
    } else {
      imag_text = std::string(remaining);
    }
  } else {
    real_text = std::string(remaining.substr(0, split));
    imag_text = std::string(remaining.substr(split));
  }
  const double real = parse_complex_literal_double(real_text, label);
  const double imag = parse_complex_literal_double(imag_text, label);
  return {static_cast<float>(real), static_cast<float>(imag)};
}

std::pair<float, float> complex64_pair_from_json(
    const OrderedJson& value,
    const std::string& label) {
  if (value.is_object() && value.contains("__mlx_complex__")) {
    const auto& pair = value.at("__mlx_complex__");
    if (!pair.is_array() || pair.size() != 2 || !json_is_numeric(pair.at(0)) ||
        !json_is_numeric(pair.at(1))) {
      throw std::invalid_argument(label + " invalid complex marker");
    }
    return {
        static_cast<float>(pair.at(0).get<double>()),
        static_cast<float>(pair.at(1).get<double>())};
  }
  if (value.is_string()) {
    return complex64_pair_from_string(
        value.get_ref<const std::string&>(), label);
  }
  if (value.is_boolean()) {
    return {value.get<bool>() ? 1.0f : 0.0f, 0.0f};
  }
  if (json_is_numeric(value)) {
    return {static_cast<float>(value.get<double>()), 0.0f};
  }
  throw std::invalid_argument(
      label + " unsupported complex64 initializer leaf");
}

template <typename IntegerType>
std::string tensor_raw_integer_initializer(
    const std::vector<const OrderedJson*>& leaves,
    size_t expected,
    const std::string& label) {
  std::string raw;
  raw.reserve(expected * sizeof(IntegerType));
  for (const auto* item : leaves) {
    append_le_bytes<IntegerType>(
        raw, static_cast<IntegerType>(normalized_integer_scalar(*item, label)));
  }
  return raw;
}

double numeric_initializer_leaf(
    const OrderedJson& value,
    const std::string& numeric_error_message) {
  if (!json_is_numeric(value)) {
    throw std::invalid_argument(numeric_error_message);
  }
  return value.get<double>();
}

template <typename FloatType>
std::string tensor_raw_float_initializer(
    const std::vector<const OrderedJson*>& leaves,
    size_t expected,
    const std::string& numeric_error_message) {
  std::string raw;
  raw.reserve(expected * sizeof(FloatType));
  for (const auto* item : leaves) {
    append_le_bytes<FloatType>(
        raw,
        static_cast<FloatType>(
            numeric_initializer_leaf(*item, numeric_error_message)));
  }
  return raw;
}

std::string tensor_raw_bool_initializer(
    const std::vector<const OrderedJson*>& leaves,
    size_t expected) {
  std::string raw;
  raw.reserve(expected);
  for (const auto* item : leaves) {
    uint8_t value = 0;
    if (item->is_boolean()) {
      value = item->get<bool>() ? 1 : 0;
    } else if (json_integer_like(*item)) {
      value = normalized_integer_scalar(*item, "bool initializer leaf") == 0
          ? 0
          : 1;
    } else if (item->is_number_float()) {
      value = item->get<double>() == 0.0 ? 0 : 1;
    } else {
      throw std::invalid_argument(
          "bool initializer values must be numeric/boolean");
    }
    raw.push_back(static_cast<char>(value));
  }
  return raw;
}

std::string tensor_raw_float16_initializer(
    const std::vector<const OrderedJson*>& leaves,
    size_t expected) {
  std::string raw;
  raw.reserve(expected * sizeof(uint16_t));
  for (const auto* item : leaves) {
    const float value = static_cast<float>(numeric_initializer_leaf(
        *item, "float16 initializer values must be numeric"));
    append_le_bytes<uint16_t>(raw, float32_to_float16_bits(value));
  }
  return raw;
}

std::string tensor_raw_bfloat16_initializer(
    const std::vector<const OrderedJson*>& leaves,
    size_t expected) {
  std::string raw;
  raw.reserve(expected * sizeof(uint16_t));
  for (const auto* item : leaves) {
    const float value = static_cast<float>(numeric_initializer_leaf(
        *item, "bfloat16 initializer values must be numeric"));
    append_le_bytes<uint16_t>(raw, float32_to_bfloat16_bits(value));
  }
  return raw;
}

std::string tensor_raw_complex64_initializer(
    const std::vector<const OrderedJson*>& leaves,
    size_t expected) {
  std::string raw;
  raw.reserve(expected * sizeof(float) * 2);
  for (const auto* item : leaves) {
    auto [real, imag] =
        complex64_pair_from_json(*item, "complex64 initializer leaf");
    append_le_bytes<float>(raw, real);
    append_le_bytes<float>(raw, imag);
  }
  return raw;
}

std::string tensor_raw_bytes_from_initializer(
    const OnnxInitializerModel& tensor) {
  const size_t expected = expected_initializer_value_count(tensor.shape);

  std::vector<const OrderedJson*> leaves;
  collect_initializer_leaves(tensor.values, leaves);
  if (leaves.size() != expected) {
    std::ostringstream out;
    out << "initializer " << tensor.name << " has " << leaves.size()
        << " values but expected " << expected;
    throw std::invalid_argument(out.str());
  }

  const std::string& dtype = tensor.dtype;
  if (dtype == "bool" || dtype == "bool_") {
    return tensor_raw_bool_initializer(leaves, expected);
  }
  if (dtype == "uint8") {
    return tensor_raw_integer_initializer<uint8_t>(
        leaves, expected, "uint8 initializer leaf");
  }
  if (dtype == "uint16") {
    return tensor_raw_integer_initializer<uint16_t>(
        leaves, expected, "uint16 initializer leaf");
  }
  if (dtype == "uint32") {
    return tensor_raw_integer_initializer<uint32_t>(
        leaves, expected, "uint32 initializer leaf");
  }
  if (dtype == "uint64") {
    return tensor_raw_integer_initializer<uint64_t>(
        leaves, expected, "uint64 initializer leaf");
  }
  if (dtype == "int8") {
    return tensor_raw_integer_initializer<int8_t>(
        leaves, expected, "int8 initializer leaf");
  }
  if (dtype == "int16") {
    return tensor_raw_integer_initializer<int16_t>(
        leaves, expected, "int16 initializer leaf");
  }
  if (dtype == "int32") {
    return tensor_raw_integer_initializer<int32_t>(
        leaves, expected, "int32 initializer leaf");
  }
  if (dtype == "int64") {
    return tensor_raw_integer_initializer<int64_t>(
        leaves, expected, "int64 initializer leaf");
  }
  if (dtype == "float16") {
    return tensor_raw_float16_initializer(leaves, expected);
  }
  if (dtype == "bfloat16") {
    return tensor_raw_bfloat16_initializer(leaves, expected);
  }
  if (dtype == "float32") {
    return tensor_raw_float_initializer<float>(
        leaves, expected, "float32 initializer values must be numeric");
  }
  if (dtype == "float64") {
    return tensor_raw_float_initializer<double>(
        leaves, expected, "float64 initializer values must be numeric");
  }
  if (dtype == "complex64") {
    return tensor_raw_complex64_initializer(leaves, expected);
  }

  throw std::invalid_argument(
      "unsupported initializer dtype for native ONNX binary export: " + dtype);
}

std::string pb_encode_string_string_entry(
    const std::string& key,
    const std::string& value) {
  std::string out;
  pb_write_string_field(out, 1, key);
  pb_write_string_field(out, 2, value);
  return out;
}

void pb_encode_tensor_header(
    std::string& out,
    const std::vector<int64_t>& shape,
    int elem_type,
    const std::string& name) {
  for (const auto dim : shape) {
    pb_write_int64_field(out, 1, dim);
  }
  pb_write_varint_field(out, 2, static_cast<uint64_t>(elem_type));
  pb_write_string_field(out, 8, name);
}

uint32_t raw_bytes_u32_le_at(const std::string& raw, size_t offset) {
  const auto b0 =
      static_cast<uint32_t>(static_cast<unsigned char>(raw[offset + 0]));
  const auto b1 =
      static_cast<uint32_t>(static_cast<unsigned char>(raw[offset + 1]));
  const auto b2 =
      static_cast<uint32_t>(static_cast<unsigned char>(raw[offset + 2]));
  const auto b3 =
      static_cast<uint32_t>(static_cast<unsigned char>(raw[offset + 3]));
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

void pb_encode_tensor_inline_complex64_data(
    std::string& out,
    const std::string& raw) {
  if ((raw.size() % sizeof(float)) != 0) {
    throw std::invalid_argument(
        "complex64 initializer raw byte count must be divisible by 4");
  }
  for (size_t offset = 0; offset < raw.size(); offset += sizeof(float)) {
    pb_write_fixed32_field(out, 4, raw_bytes_u32_le_at(raw, offset));
  }
}

void pb_encode_tensor_inline_data(
    std::string& out,
    const std::string& dtype,
    const std::string& raw) {
  if (dtype == "complex64") {
    pb_encode_tensor_inline_complex64_data(out, raw);
    return;
  }
  pb_write_bytes_field(out, 9, raw);
}

void pb_encode_tensor_external_data_entries(
    std::string& out,
    const std::string& external_data_file,
    uint64_t external_offset,
    size_t raw_size) {
  pb_write_message_field(
      out, 13, pb_encode_string_string_entry("location", external_data_file));
  pb_write_message_field(
      out,
      13,
      pb_encode_string_string_entry("offset", std::to_string(external_offset)));
  pb_write_message_field(
      out,
      13,
      pb_encode_string_string_entry("length", std::to_string(raw_size)));
  pb_write_varint_field(out, 14, 1);
}

void append_tensor_external_data(
    std::string& external_data,
    uint64_t& external_offset,
    const std::string& raw) {
  external_data.append(raw);
  external_offset += static_cast<uint64_t>(raw.size());
}

bool should_externalize_tensor_raw_bytes(
    const OnnxBinaryWriteOptions& options,
    const std::string& raw) {
  return options.external_data &&
      static_cast<int64_t>(raw.size()) >= options.external_data_size_threshold;
}

} // namespace

std::string pb_encode_tensor(
    const OnnxInitializerModel& tensor,
    const OnnxBinaryWriteOptions& options,
    std::string& external_data,
    uint64_t& external_offset,
    bool& has_external_data) {
  const std::string raw = tensor_raw_bytes_from_initializer(tensor);

  std::string out;
  pb_encode_tensor_header(out, tensor.shape, tensor.elem_type, tensor.name);

  const bool externalize = should_externalize_tensor_raw_bytes(options, raw);
  if (!externalize) {
    pb_encode_tensor_inline_data(out, tensor.dtype, raw);
    return out;
  }

  has_external_data = true;
  pb_encode_tensor_external_data_entries(
      out, options.external_data_file, external_offset, raw.size());
  append_tensor_external_data(external_data, external_offset, raw);
  return out;
}

} // namespace mlx::onnx::detail




namespace mlx::onnx::detail {
namespace {

std::string pb_encode_tensor_shape(const std::vector<int64_t>& shape) {
  std::string out;
  for (const auto dim : shape) {
    std::string dim_msg;
    pb_write_int64_field(dim_msg, 1, dim);
    pb_write_message_field(out, 1, dim_msg);
  }
  return out;
}

std::string pb_encode_tensor_type_proto(
    int elem_type,
    const std::vector<int64_t>& shape) {
  std::string tensor_type;
  pb_write_varint_field(tensor_type, 1, static_cast<uint64_t>(elem_type));
  pb_write_message_field(tensor_type, 2, pb_encode_tensor_shape(shape));

  std::string type_proto;
  pb_write_message_field(type_proto, 1, tensor_type);
  return type_proto;
}

std::string pb_encode_value_info(const OnnxValueInfoModel& info) {
  std::string out;
  pb_write_string_field(out, 1, info.name);
  pb_write_message_field(
      out, 2, pb_encode_tensor_type_proto(info.elem_type, info.shape));
  return out;
}

std::string pb_encode_attribute(
    const std::string& op_type,
    const std::string& name,
    const OrderedJson& value) {
  std::string out;
  pb_write_string_field(out, 1, name);

  if (op_type == "Cast" && name == "to" && value.is_string()) {
    const int cast_to =
        onnx_elem_type_from_symbol_lookup(value.get<std::string>());
    pb_write_varint_field(out, 20, 2);
    pb_write_int64_field(out, 3, cast_to);
    return out;
  }

  if (value.is_boolean()) {
    pb_write_varint_field(out, 20, 2);
    pb_write_int64_field(out, 3, value.get<bool>() ? 1 : 0);
    return out;
  }
  if (value.is_number_integer() || value.is_number_unsigned()) {
    pb_write_varint_field(out, 20, 2);
    pb_write_int64_field(out, 3, normalized_integer_scalar(value, "attribute"));
    return out;
  }
  if (value.is_number_float()) {
    pb_write_varint_field(out, 20, 1);
    pb_write_fixed32_field(
        out,
        2,
        std::bit_cast<uint32_t>(static_cast<float>(value.get<double>())));
    return out;
  }
  if (value.is_string()) {
    pb_write_varint_field(out, 20, 3);
    pb_write_bytes_field(out, 4, value.get<std::string>());
    return out;
  }
  if (value.is_array()) {
    bool all_integer_typed = true;
    bool all_numeric = true;
    bool all_string = true;
    for (const auto& item : value) {
      all_integer_typed = all_integer_typed &&
          (item.is_boolean() || item.is_number_integer() ||
           item.is_number_unsigned());
      all_numeric = all_numeric && json_is_numeric(item);
      all_string = all_string && item.is_string();
    }
    if (value.empty() || all_integer_typed) {
      pb_write_varint_field(out, 20, 7);
      for (const auto& item : value) {
        pb_write_int64_field(
            out, 8, normalized_integer_scalar(item, "attribute vector"));
      }
      return out;
    }
    if (all_numeric) {
      pb_write_varint_field(out, 20, 6);
      for (const auto& item : value) {
        pb_write_fixed32_field(
            out,
            7,
            std::bit_cast<uint32_t>(static_cast<float>(item.get<double>())));
      }
      return out;
    }
    if (all_string) {
      pb_write_varint_field(out, 20, 8);
      for (const auto& item : value) {
        pb_write_bytes_field(out, 9, item.get<std::string>());
      }
      return out;
    }
  }

  throw std::invalid_argument(
      "unsupported ONNX attribute type for " + op_type + "." + name);
}

std::string pb_encode_node(const OnnxNodeModel& node) {
  std::string out;
  for (const auto& input : node.inputs) {
    pb_write_string_field(out, 1, input);
  }
  for (const auto& output : node.outputs) {
    pb_write_string_field(out, 2, output);
  }
  pb_write_string_field(out, 3, node.name);
  pb_write_string_field(out, 4, node.op_type);
  for (const auto& attribute : node.attributes) {
    pb_write_message_field(
        out,
        5,
        pb_encode_attribute(node.op_type, attribute.name, attribute.value));
  }
  return out;
}

std::string pb_encode_graph(
    const OnnxGraphModel& graph,
    const OnnxBinaryWriteOptions& options,
    std::string& external_data,
    bool& has_external_data) {
  std::string out;
  pb_write_string_field(out, 2, graph.name);

  for (const auto& node : graph.nodes) {
    pb_write_message_field(out, 1, pb_encode_node(node));
  }

  uint64_t external_offset = 0;
  for (const auto& initializer : graph.initializers) {
    pb_write_message_field(
        out,
        5,
        pb_encode_tensor(
            initializer,
            options,
            external_data,
            external_offset,
            has_external_data));
  }

  for (const auto& input : graph.inputs) {
    pb_write_message_field(out, 11, pb_encode_value_info(input));
  }
  for (const auto& output : graph.outputs) {
    pb_write_message_field(out, 12, pb_encode_value_info(output));
  }

  return out;
}

std::string pb_encode_opset_import(int64_t opset) {
  std::string out;
  pb_write_string_field(out, 1, "");
  pb_write_int64_field(out, 2, opset);
  return out;
}

} // namespace

OnnxBinaryArtifact build_onnx_binary_artifact_from_stub_impl(
    const OrderedJson& onnx_stub,
    const OnnxBinaryWriteOptions& options) {
  const auto model_spec = onnx_stub_model_from_json(onnx_stub);

  std::string external_data;
  bool has_external_data = false;
  const std::string graph_message = pb_encode_graph(
      model_spec.graph, options, external_data, has_external_data);

  std::string model;
  pb_write_int64_field(model, 1, 10);
  pb_write_string_field(model, 2, model_spec.producer_name);
  pb_write_message_field(model, 7, graph_message);
  pb_write_message_field(model, 8, pb_encode_opset_import(model_spec.opset));

  OnnxBinaryArtifact artifact;
  artifact.model_bytes = std::move(model);
  artifact.external_data_bytes = std::move(external_data);
  artifact.has_external_data = has_external_data;
  return artifact;
}

} // namespace mlx::onnx::detail


namespace mlx::onnx::detail {
// Intentionally kept as a lightweight aggregation unit after splitting ONNX
// binary encoding into wire/tensor/assembly translation units.
} // namespace mlx::onnx::detail


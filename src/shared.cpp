#include "detail.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mlx::onnx::detail {
namespace {

template <size_t N>
const char* lookup_string_pair(
    const std::array<std::pair<const char*, const char*>, N>& pairs,
    const std::string& key) {
  for (const auto& [candidate_key, candidate_value] : pairs) {
    if (key == candidate_key) {
      return candidate_value;
    }
  }
  return nullptr;
}

constexpr std::array<std::pair<const char*, const char*>, 15> kOnnxDtypePairs =
    {{
        {"bool", "BOOL"},
        {"bool_", "BOOL"},
        {"uint8", "UINT8"},
        {"uint16", "UINT16"},
        {"uint32", "UINT32"},
        {"uint64", "UINT64"},
        {"int8", "INT8"},
        {"int16", "INT16"},
        {"int32", "INT32"},
        {"int64", "INT64"},
        {"float16", "FLOAT16"},
        {"float32", "FLOAT"},
        {"float64", "DOUBLE"},
        {"bfloat16", "BFLOAT16"},
        {"complex64", "COMPLEX64"},
    }};

} // namespace

std::string tagged_error_message(
    const std::string& tag,
    const std::string& message) {
  if (!message.empty() && message.front() == '[') {
    return message;
  }
  return "[" + tag + "] " + message;
}

bool json_is_numeric(const OrderedJson& value) {
  return value.is_number_integer() || value.is_number_unsigned() ||
      value.is_number_float();
}

int64_t normalized_integer_scalar(
    const OrderedJson& value,
    const std::string& label) {
  constexpr int64_t int64_min = std::numeric_limits<int64_t>::min();
  constexpr int64_t int64_max = std::numeric_limits<int64_t>::max();

  if (value.is_number_integer()) {
    return value.get<int64_t>();
  }
  if (value.is_number_unsigned()) {
    const auto raw = value.get<uint64_t>();
    if (raw <= static_cast<uint64_t>(int64_max)) {
      return static_cast<int64_t>(raw);
    }

    constexpr uint64_t uint64_max = std::numeric_limits<uint64_t>::max();
    constexpr unsigned __int128 uint64_modulus =
        static_cast<unsigned __int128>(uint64_max) + 1;
    const unsigned __int128 raw128 = static_cast<unsigned __int128>(raw);
    const __int128 wrapped =
        static_cast<__int128>(raw128) - static_cast<__int128>(uint64_modulus);
    if (wrapped >= static_cast<__int128>(int64_min) &&
        wrapped <= static_cast<__int128>(int64_max)) {
      return static_cast<int64_t>(wrapped);
    }

    std::ostringstream out;
    out << "[ir.lowering] " << label
        << " is outside supported signed 64-bit range";
    throw std::range_error(out.str());
  }
  if (value.is_number_float()) {
    const double raw = value.get<double>();
    if (!std::isfinite(raw) || std::trunc(raw) != raw) {
      std::ostringstream out;
      out << "[ir.lowering] " << label << " must be an Integer";
      throw std::invalid_argument(out.str());
    }
    if (raw < static_cast<double>(int64_min) ||
        raw > static_cast<double>(int64_max)) {
      std::ostringstream out;
      out << "[ir.lowering] " << label
          << " is outside supported signed 64-bit range";
      throw std::range_error(out.str());
    }
    return static_cast<int64_t>(raw);
  }

  std::ostringstream out;
  out << "[ir.lowering] " << label << " must be an Integer";
  throw std::invalid_argument(out.str());
}

std::vector<int64_t> normalize_integer_vector(
    const OrderedJson& value,
    const std::string& label) {
  if (value.is_array()) {
    std::vector<int64_t> out;
    out.reserve(value.size());
    for (const auto& item : value) {
      out.push_back(normalized_integer_scalar(item, label));
    }
    return out;
  }

  if (json_is_numeric(value)) {
    return {normalized_integer_scalar(value, label)};
  }

  std::ostringstream out;
  out << "[ir.lowering] " << label
      << " must be an Integer or Array of Integer";
  throw std::invalid_argument(out.str());
}

std::vector<std::string> parse_string_array(
    const OrderedJson& value,
    const std::string& label) {
  if (!value.is_array()) {
    std::ostringstream out;
    out << label << " must be an Array";
    throw std::invalid_argument(out.str());
  }

  std::vector<std::string> out;
  out.reserve(value.size());
  for (const auto& item : value) {
    if (!item.is_string()) {
      std::ostringstream msg;
      msg << label << " must contain String values";
      throw std::invalid_argument(msg.str());
    }
    out.push_back(item.get<std::string>());
  }
  return out;
}

std::string canonical_dtype(const std::string& dtype) {
  return dtype == "bool_" ? "bool" : dtype;
}

std::optional<std::string> canonical_dtype(
    const std::optional<std::string>& dtype) {
  if (!dtype.has_value()) {
    return std::nullopt;
  }
  return canonical_dtype(dtype.value());
}

std::optional<std::string> onnx_effective_dtype(
    const std::optional<std::string>& dtype) {
  if (!dtype.has_value()) {
    return std::nullopt;
  }
  const auto canonical = canonical_dtype(dtype.value());
  return canonical == "bfloat16" ? std::optional<std::string>("float32")
                                 : std::optional<std::string>(canonical);
}

std::string onnx_effective_dtype(const std::string& dtype) {
  const auto canonical = canonical_dtype(dtype);
  return canonical == "bfloat16" ? "float32" : canonical;
}

std::string onnx_dtype_symbol(const std::string& dtype) {
  const char* symbol = lookup_string_pair(kOnnxDtypePairs, dtype);
  if (symbol != nullptr) {
    return std::string(symbol);
  }

  std::ostringstream out;
  out << "[ir.lowering] unsupported dtype " << dtype;
  throw std::runtime_error(out.str());
}

void for_each_declared_payload_tensor(
    const OrderedJson& payload,
    const std::function<void(const OrderedJson&)>& visitor) {
  constexpr std::array<const char*, 3> kSections = {
      "inputs", "constants", "outputs"};
  for (const auto* section : kSections) {
    for (const auto& tensor : payload.at(section)) {
      visitor(tensor);
    }
  }
}

} // namespace mlx::onnx::detail

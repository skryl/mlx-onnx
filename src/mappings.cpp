#include "mappings.hpp"

#include <array>
#include <optional>
#include <stdexcept>
#include <string>

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

template <size_t N>
const char* lookup_int_string_pair(
    const std::array<std::pair<int64_t, const char*>, N>& pairs,
    int64_t key) {
  for (const auto& [candidate_key, candidate_value] : pairs) {
    if (candidate_key == key) {
      return candidate_value;
    }
  }
  return nullptr;
}

template <size_t N>
std::optional<int> lookup_string_int_pair(
    const std::array<std::pair<const char*, int>, N>& pairs,
    const std::string& key) {
  for (const auto& [candidate_key, candidate_value] : pairs) {
    if (key == candidate_key) {
      return candidate_value;
    }
  }
  return std::nullopt;
}

constexpr std::array<std::pair<const char*, const char*>, 54> kOnnxOpPairs = {{
    {"Add", "Add"},
    {"AddMM", "Gemm"},
    {"Subtract", "Sub"},
    {"Multiply", "Mul"},
    {"Square", "Mul"},
    {"Divide", "Div"},
    {"AsType", "Cast"},
    {"Exp", "Exp"},
    {"Log", "Log"},
    {"Sin", "Sin"},
    {"Cos", "Cos"},
    {"Erf", "Erf"},
    {"ErfInv", "ErfInv"},
    {"Sqrt", "Sqrt"},
    {"Abs", "Abs"},
    {"Floor", "Floor"},
    {"Negative", "Neg"},
    {"Relu", "Relu"},
    {"Sigmoid", "Sigmoid"},
    {"Tanh", "Tanh"},
    {"LayerNorm", "LayerNormalization"},
    {"Softmax", "Softmax"},
    {"Greater", "Greater"},
    {"Less", "Less"},
    {"Equal", "Equal"},
    {"Select", "Where"},
    {"Full", "Identity"},
    {"RandomBits", "RandomUniform"},
    {"Matmul", "MatMul"},
    {"Reshape", "Reshape"},
    {"Flatten", "Reshape"},
    {"Unflatten", "Reshape"},
    {"Transpose", "Transpose"},
    {"Squeeze", "Squeeze"},
    {"ExpandDims", "Unsqueeze"},
    {"Broadcast", "Expand"},
    {"Arange", "Constant"},
    {"AsStrided", "Gather"},
    {"RoPE", "Identity"},
    {"Concatenate", "Concat"},
    {"Convolution", "Conv"},
    {"ConvolutionTranspose", "ConvTranspose"},
    {"Gather", "Gather"},
    {"GatherAxis", "GatherElements"},
    {"Slice", "Slice"},
    {"SliceUpdate", "ScatterND"},
    {"Split", "Split"},
    {"LogSumExp", "ReduceLogSumExp"},
    {"Pad", "Pad"},
    {"Scan", "CumSum"},
    {"ScatterAxis", "ScatterElements"},
    {"Maximum", "Max"},
    {"Minimum", "Min"},
    {"Power", "Pow"},
}};

constexpr std::array<std::pair<int64_t, const char*>, 6> kReduceCodeToOnnxOp = {
    {
        {0, "ReduceMin"},
        {1, "ReduceMax"},
        {2, "ReduceSum"},
        {3, "ReduceProd"},
        {4, "ReduceMin"},
        {5, "ReduceMax"},
    }};

constexpr std::array<std::pair<int64_t, const char*>, 2>
    kArgReduceCodeToOnnxOp = {{
        {0, "ArgMin"},
        {1, "ArgMax"},
    }};

constexpr std::array<std::pair<const char*, int>, 13> kDtypePromotionRankPairs =
    {{
        {"bool", 0},
        {"uint8", 1},
        {"int8", 2},
        {"uint16", 3},
        {"int16", 4},
        {"uint32", 5},
        {"int32", 6},
        {"uint64", 7},
        {"int64", 8},
        {"bfloat16", 9},
        {"float16", 10},
        {"float32", 11},
        {"float64", 12},
    }};

constexpr std::array<std::pair<const char*, int>, 17> kOnnxElemTypePairs = {{
    {"UNDEFINED", 0},
    {"FLOAT", 1},
    {"UINT8", 2},
    {"INT8", 3},
    {"UINT16", 4},
    {"INT16", 5},
    {"INT32", 6},
    {"INT64", 7},
    {"STRING", 8},
    {"BOOL", 9},
    {"FLOAT16", 10},
    {"DOUBLE", 11},
    {"UINT32", 12},
    {"UINT64", 13},
    {"COMPLEX64", 14},
    {"COMPLEX128", 15},
    {"BFLOAT16", 16},
}};

} // namespace

std::optional<std::string> onnx_op_type_for_ir_op(const std::string& op) {
  const char* mapped = lookup_string_pair(kOnnxOpPairs, op);
  if (mapped == nullptr) {
    return std::nullopt;
  }
  return std::string(mapped);
}

std::optional<std::string> onnx_reduce_op_type_from_code(int64_t reduce_code) {
  const char* mapped = lookup_int_string_pair(kReduceCodeToOnnxOp, reduce_code);
  if (mapped == nullptr) {
    return std::nullopt;
  }
  return std::string(mapped);
}

std::optional<std::string> onnx_argreduce_op_type_from_code(
    int64_t reduce_code) {
  const char* mapped =
      lookup_int_string_pair(kArgReduceCodeToOnnxOp, reduce_code);
  if (mapped == nullptr) {
    return std::nullopt;
  }
  return std::string(mapped);
}

std::optional<int> dtype_promotion_rank(const std::string& dtype) {
  return lookup_string_int_pair(kDtypePromotionRankPairs, dtype);
}

int onnx_elem_type_from_symbol_lookup(const std::string& symbol) {
  const auto mapped = lookup_string_int_pair(kOnnxElemTypePairs, symbol);
  if (mapped.has_value()) {
    return mapped.value();
  }
  throw std::invalid_argument(
      "unsupported ONNX element type symbol: " + symbol);
}

} // namespace mlx::onnx::detail

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "json.hpp"

namespace mlx::onnx::detail {

using Shape = std::vector<int64_t>;
using ShapeMap = std::map<std::string, Shape>;
using DtypeMap = std::map<std::string, std::string>;
using NameSet = std::set<std::string>;

std::string tagged_error_message(
    const std::string& tag,
    const std::string& message);
bool json_is_numeric(const OrderedJson& value);
int64_t normalized_integer_scalar(
    const OrderedJson& value,
    const std::string& label);
std::vector<int64_t> normalize_integer_vector(
    const OrderedJson& value,
    const std::string& label);
std::vector<std::string> parse_string_array(
    const OrderedJson& value,
    const std::string& label);
std::string canonical_dtype(const std::string& dtype);
std::optional<std::string> canonical_dtype(
    const std::optional<std::string>& dtype);
std::optional<std::string> onnx_effective_dtype(
    const std::optional<std::string>& dtype);
std::string onnx_effective_dtype(const std::string& dtype);
std::string onnx_dtype_symbol(const std::string& dtype);
void for_each_declared_payload_tensor(
    const OrderedJson& payload,
    const std::function<void(const OrderedJson&)>& visitor);

struct LoweringContext {
  OrderedJson& initializers;
  NameSet& used_tensor_names;
  ShapeMap& known_shapes;
  DtypeMap& known_dtypes;
};

NameSet collect_payload_tensor_names(const OrderedJson& payload);
ShapeMap collect_known_tensor_shapes(const OrderedJson& payload);
DtypeMap collect_known_tensor_dtypes(const OrderedJson& payload);
std::optional<std::string> onnx_op_type_for_node(
    const OrderedJson& node,
    bool strict,
    const ShapeMap* known_shapes);
std::vector<OrderedJson> lower_onnx_node_default(
    const OrderedJson& node,
    size_t node_index,
    LoweringContext& lowering);
OrderedJson ir_compatibility_report_payload_impl(
    const OrderedJson& payload);
OnnxBinaryArtifact build_onnx_binary_artifact_from_stub_impl(
    const OrderedJson& onnx_stub,
    const OnnxBinaryWriteOptions& options);

} // namespace mlx::onnx::detail

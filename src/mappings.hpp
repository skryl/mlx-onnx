#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace mlx::onnx::detail {

std::optional<std::string> onnx_op_type_for_ir_op(const std::string& op);
std::optional<std::string> onnx_reduce_op_type_from_code(int64_t reduce_code);
std::optional<std::string> onnx_argreduce_op_type_from_code(
    int64_t reduce_code);
std::optional<int> dtype_promotion_rank(const std::string& dtype);
int onnx_elem_type_from_symbol_lookup(const std::string& symbol);

} // namespace mlx::onnx::detail

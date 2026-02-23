#include "detail.hpp"

#include <exception>
#include <limits>
#include <optional>
#include <set>
#include <utility>

namespace mlx::onnx::detail {

OrderedJson ir_compatibility_report_payload_impl(
    const OrderedJson& payload) {
  // Simulate lowering node-by-node on cloned state. Successful probes commit
  // inferred state so later nodes see realistic tensor facts.
  OrderedJson probe_initializers = OrderedJson::array();
  auto probe_used_tensor_names = collect_payload_tensor_names(payload);
  auto probe_known_shapes = collect_known_tensor_shapes(payload);
  auto probe_known_dtypes = collect_known_tensor_dtypes(payload);

  OrderedJson node_support = OrderedJson::array();
  size_t unsupported_nodes = 0;
  std::set<std::string> unsupported_ops;

  const auto& source_nodes = payload.at("nodes");
  for (size_t index = 0; index < source_nodes.size(); ++index) {
    const auto& node = source_nodes.at(index);
    const auto op = node.at("op").get<std::string>();

    bool supported = false;
    std::optional<std::string> mapped;

    try {
      auto trial_initializers = probe_initializers;
      auto trial_used_tensor_names = probe_used_tensor_names;
      auto trial_known_shapes = probe_known_shapes;
      auto trial_known_dtypes = probe_known_dtypes;
      LoweringContext trial_lowering{
          trial_initializers,
          trial_used_tensor_names,
          trial_known_shapes,
          trial_known_dtypes};

      auto lowered = lower_onnx_node_default(node, index, trial_lowering);
      if (!lowered.empty() && lowered.front().contains("op_type") &&
          lowered.front().at("op_type").is_string()) {
        mapped = lowered.front().at("op_type").get<std::string>();
      } else {
        mapped = onnx_op_type_for_node(node, false, &trial_known_shapes);
      }

      probe_initializers = std::move(trial_initializers);
      probe_used_tensor_names = std::move(trial_used_tensor_names);
      probe_known_shapes = std::move(trial_known_shapes);
      probe_known_dtypes = std::move(trial_known_dtypes);
      supported = true;
    } catch (const std::exception&) {
      try {
        mapped = onnx_op_type_for_node(node, false, &probe_known_shapes);
      } catch (const std::exception&) {
        mapped = std::nullopt;
      }
    }

    OrderedJson entry = OrderedJson::object();
    entry["index"] = index;
    entry["op"] = op;
    entry["supported"] = supported;
    if (mapped.has_value()) {
      entry["onnx_op_type"] = mapped.value();
    } else {
      entry["onnx_op_type"] = nullptr;
    }
    node_support.push_back(std::move(entry));

    if (!supported) {
      ++unsupported_nodes;
      unsupported_ops.insert(op);
    }
  }

  int64_t ir_version = kGraphIrVersion;
  if (payload.contains("ir_version")) {
    const auto& value = payload.at("ir_version");
    if (value.is_number_integer()) {
      ir_version = value.get<int64_t>();
    } else if (value.is_number_unsigned()) {
      const auto raw = value.get<uint64_t>();
      if (raw <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        ir_version = static_cast<int64_t>(raw);
      }
    }
  }

  OrderedJson unsupported_ops_json = OrderedJson::array();
  for (const auto& unsupported_op : unsupported_ops) {
    unsupported_ops_json.push_back(unsupported_op);
  }

  OrderedJson report = OrderedJson::object();
  report["format"] = "webgpu_compat_report_v1";
  report["ir_version"] = ir_version;
  report["total_nodes"] = source_nodes.size();
  report["supported_nodes"] = source_nodes.size() - unsupported_nodes;
  report["unsupported_nodes"] = unsupported_nodes;
  report["unsupported_ops"] = std::move(unsupported_ops_json);
  report["ready_for_stub_conversion"] = unsupported_nodes == 0;
  report["nodes"] = std::move(node_support);
  return report;
}

} // namespace mlx::onnx::detail

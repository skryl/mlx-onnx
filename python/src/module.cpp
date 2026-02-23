#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_onnx(nb::module_&);

NB_MODULE(_core, m) {
  m.doc() = "mlx-onnx IR bindings";
  init_onnx(m);
}

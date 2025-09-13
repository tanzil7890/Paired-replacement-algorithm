#include <torch/extension.h>
#include "paired/weight_cache.h"

// Register custom class in namespace "paired"
TORCH_LIBRARY(paired, m) {
  m.class_<WeightCache>("WeightCache")
      .def(torch::init<const torch::Tensor&, int64_t,
                       const torch::Tensor&, const torch::Tensor&,
                       const torch::Tensor&, bool>())
      .def("update_active_weights", &WeightCache::update_active_weights)
      .def("get_concat_weight", &WeightCache::get_concat_weight)
      .def("get_active_down_weight", &WeightCache::get_active_down_weight)
      .def("get_num_active", &WeightCache::get_num_active);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Intentionally left empty; class is registered via TORCH_LIBRARY.
}

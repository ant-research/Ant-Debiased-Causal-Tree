#include <pybind11/pybind11.h>

#include "include/did_utils.h"
#include "include/gbct_utils.h"
#include "include/utils/nparray_ops.hpp"

namespace py = pybind11;


void gbct_utils::DebiasNode::set_outcomes(const py::array_t<double>& value){ 
    outcomes = value;
}

inline const py::array_t<double>& gbct_utils::DebiasNode::get_info(const std::string &name) const{
  if (has(name) == false) {
    LightGBM::Log::Fatal("The `name`(%s) is not included in DiDNode", name.c_str());
  }
  if (name == "outcomes"){
    return outcomes;
  } else if(name == "bias"){
    return bias;
  } else if (name == "eta"){
    return eta;
  } else if (name == "debiased_effect"){
    return debiased_effect;
  } else if (name == "effect"){
    return effect;
  }
  if (infos.find(name) == infos.end()){
    LightGBM::Log::Fatal("The key `%s` is out of range", name.c_str());
  }
  return infos.at(name);
}

int gbct_utils::DebiasNode::set_info(const std::string &name, const py::array_t<double> &value) {
  if (has(name) == false) {
    LightGBM::Log::Fatal("The `name`(%s) is not included in DiDNode", name.c_str());
    return -1;
  }
  if (name == "outcomes") {
    outcomes = value;
    // insert `effect`
    std::vector<int> shape(outcomes.shape(), outcomes.shape()+outcomes.ndim());
    infos["effect"] = py::array_t<double>(shape);
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        mutable_at_(infos["effect"], i, j) = at_(outcomes, i, j) - at_(outcomes, i, 0);
      }
    }
  } else if (name == "bias") {
    bias = value;
  } else if (name == "eta") {
    eta = value;
  } else if (name == "debiased_effect"){
    debiased_effect = value;
  } else if (name == "effect"){
    effect = value;
  }

  infos[name] = value;
  return 0;
}


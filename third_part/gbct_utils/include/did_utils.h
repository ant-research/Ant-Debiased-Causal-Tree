#ifndef __DID_HEADER__
#define __DID_HEADER__

#include <cmath>
#include <iostream>
#include <set>
#include <unordered_map>

#include <omp.h>
#include <pybind11/numpy.h>
// #include <pybind11/eigen.h>

#include "include/utils.h"
#include "utils/nparray_ops.hpp"

namespace py = pybind11;

namespace gbct_utils {
class Node {
  std::set<std::string> _info_keys = {"split_feature", "split_thresh", "leaf_id", "level_id", "children", "is_leaf"};

public:
  int split_feature;
  double split_thresh;
  int leaf_id;
  int level_id;
  int children[2];
  bool is_leaf;
  void set_children(const std::vector<int> &_ch) {
    memcpy(children, _ch.data(), 2 * sizeof(int));
  }
  inline int get_split_feature() const { return split_feature; }
  inline double get_split_thresh() const { return split_thresh; }
  inline int get_leaf_id() const { return leaf_id; }
  inline int get_level_id() const { return level_id; }
  inline int leaf() const { return is_leaf; }
  std::vector<int> get_children() const {
    return std::vector<int>(children, children + 2);
  }
  virtual const py::array_t<double> &
  get_info(const std::string &name) const = 0;
  virtual int set_info(const std::string &name,
                       const py::array_t<double> &) = 0;
  virtual bool has(const std::string &key) const {
    return _info_keys.find(key) != _info_keys.end();
  }
  virtual std::vector<std::string> get_property_keys() const {
    return std::vector<std::string>(_info_keys.begin(), _info_keys.end());
  }
};

class GradientNode {
public:
  virtual py::array_t<double> estimate(const py::array_t<double> &G,const py::array_t<double> &H, double lambd) {
    return G / (H + lambd);
  }
  virtual py::array_t<double>
  approx_loss(const py::array_t<double> y_hat, const py::array_t<double> &grad,
              const py::array_t<double> &hess, double lambd = 0,
              const py::array_t<double> &weight = py::array_t<double>()) {
    if (empty_array(y_hat) == true) {
      return -.5 * (grad * grad) / (hess + lambd);
    }
    return grad * y_hat + .5 * (y_hat * y_hat) * (hess + lambd);
  }
};

class DebiasNode : public Node, public GradientNode {
protected:
  std::set<std::string> _info_keys = {"outcomes", "bias", "effect", "eta", "debiased_effect"};
  std::unordered_map<std::string, py::array_t<double>> infos;

public:
  py::array_t<double> bias;
  py::array_t<double> outcomes;
  py::array_t<double> effect;
  py::array_t<double> eta;
  py::array_t<double> debiased_effect;
  virtual const py::array_t<double> &get_outcomes() const { return outcomes; }
  virtual void set_outcomes(const py::array_t<double> &);
  virtual bool has(const std::string &key) const {
    return (this->Node::has(key)) || (_info_keys.find(key) != _info_keys.end());
  }
  virtual const py::array_t<double> &get_info(const std::string &name) const;
  virtual int set_info(const std::string &name, const py::array_t<double> &);
  virtual std::vector<std::string> get_property_keys() const {
    auto tmp = this->Node::get_property_keys();
    tmp.insert(tmp.end(), _info_keys.begin(), _info_keys.end());
    return tmp;
  }
};

class DiDNode : public DebiasNode {
  std::set<std::string> _info_keys = {"outcomes", "bias", "zz", "zy"};

public:
  virtual bool has(const std::string &key) const {
    return (this->DebiasNode::has(key)) ||
           (_info_keys.find(key) != _info_keys.end());
  }
  virtual std::vector<std::string> get_property_keys() const {
    auto tmp = this->DebiasNode::get_property_keys();
    tmp.insert(tmp.end(), _info_keys.begin(), _info_keys.end());
    return tmp;
  }
};

} // namespace gbct_utils
#endif // __DID_HEADER__
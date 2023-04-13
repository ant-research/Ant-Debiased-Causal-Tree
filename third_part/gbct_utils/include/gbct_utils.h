#ifndef __GBCT_UTILS_H__
#define __GBCT_UTILS_H__
#include <cmath>
#include <iostream>

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "utils.h"
#include "utils/log.h"
#include "utils/nparray_ops.hpp"


namespace py = pybind11;
namespace gbct_utils{
/**
 * @brief
 *
 * @tparam idtype
 * @param x_binned [n, n_features]. The binned features
 * @param insidx [n]. The index of each instance 
 * @param split_infos [L, 2]. The splitting information of `L` tree nodes, like <feature, threshold>. 
 * @param scope [L, 2]. The range of each tree node. 
 * @param pos_mid [n]
 */
template <typename idtype>
void update_x_map(const py::array_t<idtype>& x_binned,
                  py::array_t<idtype>& insidx,
                  const py::array_t<idtype>& split_infos,
                  const py::array_t<idtype>& scope,
                  py::array_t<idtype>& pos_mid) {
    // check shape
    int n = x_binned.shape(0);
    int l = split_infos.shape(0);
    shape_match(x_binned, insidx);
    shape_match(split_infos, scope);

    std::vector<idtype> out(insidx.size());
    for (int i = 0; i < l; i++) {
        auto split_feature = split_infos.at(i, 0);
        auto split_value = split_infos.at(i, 1);
        auto pos_st = scope.at(i, 0);
        auto pos_end = scope.at(i, 1);
        auto left_cur = pos_st, right_cur = pos_end - 1;

        for (idtype j = pos_st; j < pos_end; j++) {
            if (x_binned.at(insidx.at(j), split_feature) <= split_value) {
                out[left_cur++] = insidx.at(j);
            } else {
                out[right_cur--] = insidx.at(j);
            }
        }
        pos_mid.mutable_at(i << 1, 0) = pos_st;
        pos_mid.mutable_at(i << 1, 1) = left_cur;
        pos_mid.mutable_at((i << 1) + 1, 0) = left_cur;
        pos_mid.mutable_at((i << 1) + 1, 1) = pos_end;
        memcpy(insidx.mutable_data() + pos_st, out.data() + pos_st, (pos_end - pos_st) * sizeof(idtype));
    }
}

template <typename dtype, typename idtype>
py::array_t<dtype>& update_histogram(const py::array_t<dtype>& target,
                                     const py::array_t<idtype>& x_binned,
                                     const py::array_t<idtype>& index,
                                     const py::array_t<idtype>& leaves_range,
                                     const py::array_t<idtype>& treatment,
                                     py::array_t<dtype>& out,
                                     int n_treatment = 2,
                                     int n_bins = 64,
                                     int threads = 20) {
    auto n = target.shape(0), m = x_binned.shape(1), l = leaves_range.shape(0);
    int n_y = target.shape(1), n_w = n_treatment;
    py::gil_scoped_release release;
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (size_t fid = 0; fid < m; fid++) {
        for (size_t leaf_id = 0; leaf_id < l; leaf_id++) {
            // from leaves_range[leaf_id, 0] to leaves_range[leaf_id, 1]
            auto pos_st = leaves_range.at(leaf_id, 0), pos_end = leaves_range.at(leaf_id, 1);
            for (size_t _idx = pos_st; _idx < pos_end; _idx++) {
                auto id = index.at(_idx);
                auto bin_value = x_binned.at(id, fid), w_id = treatment.at(id);
                for (size_t y_id = 0; y_id < n_y; y_id++) {
                    out.mutable_at(leaf_id, fid, bin_value, w_id, y_id) += target.at(id, y_id);
                }
            }
        }
    }
    return out;
}

template <typename dtype, typename idtype>
py::array_t<double> calculate_loss(
      const std::unordered_map<int, std::unordered_map<int, std::vector<int>>>&configs,
      const py::array_t<dtype> &bin_grad_hist,
      const py::array_t<dtype> &bin_hess_hist,
      const py::array_t<dtype> &bin_cgrad_hist,
      const py::array_t<dtype> &bin_chess_hist,
      const py::array_t<idtype> &bin_counts, dtype lambd, dtype coeff, int t0){
  // [n_leaf, n_features, n_bins, n_treatment, n_outcome]
  auto shape = bin_grad_hist.shape();
  auto Gs = sum(bin_grad_hist, 2);
  auto Hs = sum(bin_hess_hist, 2);
  auto CGs = sum(bin_cgrad_hist, 2);
  auto CHs = sum(bin_chess_hist, 2);
  auto Cnt = sum(bin_counts, 2);

  auto est_fn = [](const py::array_t<dtype> &g,
                   const py::array_t<dtype> &h,
                   double lambd)->py::array_t<dtype> { return -g / (h + lambd); };
  auto get_eta = [](){};
  auto loss_fn = [](const py::array_t<dtype> &y_hat,
                    const py::array_t<dtype> &grad,
                    const py::array_t<dtype> &hess,
                    double lambd = 0) -> py::array_t<dtype> {
    return grad * y_hat + .5 * (y_hat * y_hat) * (hess + lambd);
  };
  int n_leafs = shape[0], n_feats = shape[1], n_bins = shape[2], n_treats = shape[3], n_outs = shape[4];
  // [n_treatment, n_outcome]
  py::array_t<dtype> l_grad({n_treats, n_outs}), l_hess({n_treats, n_outs}),
      l_cgrad({n_treats, n_outs}), l_chess({n_treats, n_outs}),
      r_grad({n_treats, n_outs}), r_hess({n_treats, n_outs}),
      r_cgrad({n_treats, n_outs}), r_chess({n_treats, n_outs});

  py::array_t<idtype> l_cnt({n_treats, 1}), r_cnt({n_treats, 1});
  LightGBM::Log::Info("Begin for~");
  for (auto it = configs.begin(); it != configs.end(); ++it){
      int level_id = it->first;
      for(auto fit=it->second.begin(); fit != it->second.end(); ++fit){
          int fid = fit->first;
          auto cur_bin_idx = 0;
          for (ssize_t b = 0; b < n_bins; ++b) {
            for (size_t i = 0; i < n_treats; i++){
                l_cnt.mutable_at(i, 0) += bin_counts.at(level_id, fid, b, i);
                for (size_t j = 0; j < n_outs; j++){
                    l_grad.mutable_at(i, j) += bin_grad_hist.at(level_id, fid, b, i, j);
                    l_hess.mutable_at(i, j) += bin_hess_hist.at(level_id, fid, b, i, j);
                    l_cgrad.mutable_at(i, j) += bin_cgrad_hist.at(level_id, fid, b, i, j);
                    l_chess.mutable_at(i, j) += bin_chess_hist.at(level_id, fid, b, i, j);
                    // 
                    r_grad.mutable_at(i, j) = Gs.at(level_id, fid, i, j) - l_grad.at(i, j);
                    r_hess.mutable_at(i, j) = Hs.at(level_id, fid, i, j) - l_hess.at(i, j);
                    r_cgrad.mutable_at(i, j) = CGs.at(level_id, fid, i, j) - l_cgrad.at(i, j);
                    r_chess.mutable_at(i, j) = CHs.at(level_id, fid, i, j) - l_chess.at(i, j);
                }
            }
            py::array_t<dtype> l_grad_hat({n_treats, n_outs}), l_hess_hat({n_treats, n_outs}), 
                    r_grad_hat({n_treats, n_outs}), r_hess_hat({n_treats, n_outs});
            if (cur_bin_idx < fit->second.size() && fit->second[cur_bin_idx] == b){
                cur_bin_idx += 1;
                  // left node parameter estimation
                LightGBM::Log::Info("(%d, %d) (%d, %d)", l_grad.shape(0),
                                    l_grad.shape(1), l_hess.shape(0),
                                    l_hess.shape(1));
                auto l_f_yhat = est_fn(l_grad, l_hess, lambd);
                LightGBM::Log::Info("2(%d, %d) (%d, %d)", (sum(l_grad, 0, true) - l_grad).shape(0),
                                    (sum(l_grad, 0, true) - l_grad).shape(1), l_hess.shape(0),
                                    l_hess.shape(1));
                auto l_c_yhat = est_fn(sum(l_grad, 0, true) - l_grad, sum(l_hess, 0, true) - l_hess, lambd);
                auto r_f_yhat = est_fn(r_grad, r_hess, lambd);
                auto r_c_yhat = est_fn(sum(r_grad, 0, true) - r_grad, sum(r_hess, 0, true) - r_hess, lambd);
                for (size_t w = 0; w < n_treats; w++){
                    r_cnt.mutable_at(w, 0) = Cnt.at(level_id, 0, w, 0) - l_cnt.at(w, 0);
                    for (size_t t = 0; t < n_outs; t++){
                        if (t < t0){
                            l_grad_hat.mutable_at(w, t) = l_cgrad.at(w, t);
                            l_hess_hat.mutable_at(w, t) = l_chess.at(w, t);
                            r_grad_hat.mutable_at(w, t) = r_cgrad.at(w, t);
                            r_hess_hat.mutable_at(w, t) = r_chess.at(w, t);
                        }else{
                            l_c_yhat.mutable_at(w, t) = l_f_yhat.at(w, t);
                            l_grad_hat.mutable_at(w, t) = l_grad.at(w, t);
                            l_hess_hat.mutable_at(w, t) = l_hess.at(w, t);
                            r_c_yhat.mutable_at(w, t) = r_f_yhat.at(w, t);
                            r_grad_hat.mutable_at(w, t) = r_grad.at(w, t);
                            r_hess_hat.mutable_at(w, t) = r_hess.at(w, t);
                        }
                    }   
                }
                // right node parameter estimation
                auto lloss = loss_fn(l_c_yhat, l_grad_hat, l_hess_hat, lambd);
                auto rloss = loss_fn(r_c_yhat, r_grad_hat, r_hess_hat, lambd);
                dtype loss = 0;
                
                for (size_t t = 0; t < n_outs; t++){
                    dtype temp = 0;
                    for (ssize_t w=0; w < n_treats; w++){
                        temp += lloss.at(w, t)/l_cnt.at(w, 0) * sum(l_cnt);
                        temp += rloss.at(w, t)/r_cnt.at(w, 0) * sum(r_cnt);
                    }
                    if (t < t0){
                        loss += temp / t0;
                    }else{
                        loss += coeff * temp / (n_outs - t0);
                    }
                }
            }
          }
      }
  }
}
} // namespace gbct_utils
#endif  // __GBCT_UTILS_H__
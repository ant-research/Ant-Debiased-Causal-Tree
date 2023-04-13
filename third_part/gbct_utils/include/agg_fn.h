#ifndef __AGGFN_HEADER__
#define __AGGFN_HEADER__
#include <iostream>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include "include/utils.h"
#include "utils/log.h"
#include "utils/threading.h"
#include "utils/common.h"

namespace py = pybind11;


namespace gbct_utils
{

// 定义转换函数
template <typename dtype>
dtype identity(dtype x) {
    return x;
}

template <typename dtype, typename idtype, typename std::enable_if<std::is_integral<idtype>::value>::type* = nullptr>
void sumby_parallel(const py::array_t<dtype> &data // [n, T]
        , const py::array_t<idtype> &treatment // [n, ] 
        , const py::array_t<idtype> &bin_x // [n, m]
        , std::unordered_map<size_t, py::array_t<dtype>> &outs //[m, n_treatment, n_bin, ny]
        , std::string& transform
        , int num_threads=-1) {
    std::function<dtype(dtype)> transform_func;
    if("log" == transform){
        transform_func = [](dtype x){return std::log(x);};
    }else if ("square" == transform){
        transform_func = [](dtype x){return x*x;};
    }else if ("identity" == transform){
        transform_func = [] (dtype x){return x;};
    }else{
        throw std::runtime_error("Unknown transform function'"+transform+"'\n");
    }
    // checking shape match
    size_t n_ins = data.shape(0);
    size_t n_t = data.ndim() > 1 ? data.shape(1) : 1;
    size_t n_f = bin_x.shape(1);
    int n_treatment=-1;
    // openmp not support parallel for unordered_map
    std::vector<py::array_t<dtype>*> tmp_outs;
    for (size_t i = 0; i < outs.size(); i++){
        tmp_outs.push_back(&outs[i]);
        memset(outs[i].mutable_data(), 0, outs[i].size() * sizeof(dtype));// set zeros
    }
    
    if (!shape_match(data, treatment) || !shape_match(data, bin_x)){
        std::string msg = "The instance of `data`, `treatment` and `bin_x` doesn't match!"; 
        LightGBM::Log::Fatal(msg.c_str());
    }
    if (outs.size() != n_f){
        LightGBM::Log::Fatal("The size of `outs`(%d) not equals features(%d) ", outs.size(), n_f);
    }
    // checking out shape
    for (int i = 0; i < outs.size(); ++i){
        if (n_treatment < 0){
            n_treatment = outs[i].shape(0);
        }
        if (n_treatment != outs[i].shape(0) || (outs[i].ndim() == 3 && outs[i].shape(2) != n_t) 
            || (outs[i].ndim() == 2 && n_t != 1) || (outs[i].ndim() == 1)){
            LightGBM::Log::Fatal("the shape of outs[%d] doesn't match, n_treatment or time", i);
        }
    }
    py::gil_scoped_release release;
    OMP_SET_NUM_THREADS(num_threads);
#pragma omp parallel for
    for (int f = 0; f < tmp_outs.size(); ++f){// travel for all features
        auto &out = *tmp_outs[f];
        for (size_t i = 0; i < n_ins; i++){ // travel for all instances
            auto col = treatment.at(i, 0), row = bin_x.at(i, f);
            for (size_t j = 0; j < n_t; ++j){ // travel for all time step of outcome
                auto tmp = transform_func(data.ndim() > 1 ? data.at(i, j):data.at(i));
                if (out.ndim() == 3)
                    out.mutable_at(col, row, j) += tmp;
                else
                    out.mutable_at(col, row) += tmp;
            }
        }
    }
}


template <typename dtype, typename idtype, typename std::enable_if<std::is_integral<idtype>::value>::type* = nullptr>
py::array_t<dtype>& sumby_2d(const py::array_t<dtype> &data
        , const py::array_t<idtype> &by1
        , const py::array_t<idtype> &by2
        , py::array_t<dtype> &out
        , std::function<dtype(dtype)> transform = identity<dtype>) {
    
    if (data.ndim() != 2 || by1.ndim() != 1 || by2.ndim() != 1){
        std::string msg = "data("+std::to_string(data.ndim())+"), by1("+std::to_string(by1.ndim())+"), "
            "by2("+std::to_string(by2.ndim())+") must be 1-d arrays"; 
        std::cerr<<msg<<std::endl;
        throw std::runtime_error(msg);
    }
    if (out.ndim() != 3){
        std::string msg = "The parameter `out` must be 3-d array, acutally " + std::to_string(out.ndim());
        std::cerr<<msg<<std::endl;
        throw std::runtime_error(msg);
    }
    if (out.shape()[2] != data.shape()[1]){
        std::string msg = "The last dimension of out (" + std::to_string(out.shape()[2]) + ") and data("\
            +std::to_string(data.shape()[1])+") must be the same!";
        std::cerr<<msg<<std::endl;
        throw std::runtime_error(msg);
    }
    auto shape = out.shape();
    auto uc_data = data.unchecked();
    auto uc_by1 = by1.unchecked();
    auto uc_by2 = by2.unchecked();
    auto uc_out = out.mutable_unchecked();
    if (data.shape()[0] != by1.shape()[0] || data.shape()[0] != by2.shape()[0]){
        throw std::runtime_error("Input shapes must match");
    }
    
    py::gil_scoped_release release;
    for (size_t i = 0; i < uc_data.shape(0); i++){
        auto col = uc_by1(i), row = uc_by2(i);
        if (row < 0){
            LightGBM::Log::Warning("data(%d, %d(%d))", col, row, by2.at(i));
        }
        for (size_t j = 0; j < data.shape(1); j++){
            uc_out(col,row, j) += transform(uc_data(i, j));
        }
    }
    return out;
}


template <typename dtype, typename idtype, typename std::enable_if<std::is_integral<idtype>::value>::type* = nullptr>
py::array_t<dtype>& sumby(const py::array_t<dtype, 1> &data
        , const py::array_t<idtype, 1> &by1
        , const py::array_t<idtype, 1> &by2
        , py::array_t<dtype> &out
        , std::function<dtype(dtype)> transform = identity<dtype>){

    if (data.ndim() != 1 || by1.ndim() != 1 || by2.ndim() != 1){
        std::string msg = "data("+std::to_string(data.ndim())+"), by1("+std::to_string(by1.ndim())+"), "
            "by2("+std::to_string(by2.ndim())+") must be 1-d arrays"; 
        throw std::runtime_error(msg);
    }

    auto shape = out.shape();
    auto uc_data = data.unchecked();
    auto uc_by1 = by1.unchecked();
    auto uc_by2 = by2.unchecked();
    auto uc_out = out.mutable_unchecked();
    if (uc_data.shape(0) != uc_by1.shape(0) || uc_data.shape(0) != uc_by2.shape(0)){
        std::string msg = "data("+std::to_string(data.shape(0))+"), by1("+std::to_string(by1.shape(0))+"), "
            "by2("+std::to_string(by2.shape(0))+") must be the same length"; 
        throw std::runtime_error(msg);
    }

    {
        py::gil_scoped_release release;
        for (size_t i = 0; i < data.shape()[0]; i++){
            auto col = uc_by1(i), row = uc_by2(i);
            uc_out(col,row) += transform(uc_data(i));
        }
    }
    return out;
}


template <typename dtype, typename idtype, typename std::enable_if<std::is_integral<idtype>::value>::type* = nullptr>
py::array_t<idtype> & countby(const py::array_t<dtype> &data
        , std::vector<py::array_t<idtype>> &by
        , py::array_t<idtype> &out
        , bool dropna=true){
    auto shape = out.shape();
    auto n_by = by.size();
    auto uc_data = data.unchecked();
    auto uc_out = out.mutable_unchecked();
    auto out_ptr = out.mutable_data();
    std::vector<const idtype*> by_ptrs(n_by);
    std::vector<ssize_t> strides(n_by, 1);
    for (size_t i = 0; i < n_by; i++){
        by_ptrs[i] = by[i].data();
        for (size_t j = i+1; j < n_by; j++){
            strides[i] *= shape[j];
        }
    }

    py::gil_scoped_release release;
    for (size_t i = 0; i < data.shape()[0]; i++){
        auto step = 0;
        for (size_t j = 0; j < n_by; j++){
            idtype value = by_ptrs[j][i];
            step += strides[j]*value;
        }
        int cnt = 0;
        if (dropna == true){
            if(uc_data.ndim() == 1 && std::isnan(uc_data(i)) != true)
                cnt = 1;
            else if(uc_data.ndim() == 2){
                for (size_t j = 0; j < uc_data.shape(1); j++){
                    if(std::isnan(uc_data(i, j)) != true){
                        cnt = 1;
                        break;
                    }
                }
            }
        }else{
            cnt = 1;
        }
        // uc_out(col, row) += cnt;
        out_ptr[step] += cnt;
    }
    return out;
}

template <typename dtype, typename idtype, typename std::enable_if<std::is_integral<idtype>::value>::type* = nullptr>
void countby_parallel(const py::array_t<dtype> &data // [n, T]
        , const py::array_t<idtype> &treatment // [n, ] 
        , const py::array_t<idtype> &bin_x // [n, m]
        , std::unordered_map<int, py::array_t<idtype>> &outs //{feature: [n_treatment, n_bin]}
        , bool dropna=true
        , int num_threads=-1){
    // checking shape match
    size_t n_ins = data.shape(0), n_t = (data.ndim() == 2 ? data.shape(1) : 1) ;
    size_t n_f = bin_x.shape(1);
    int n_treatment=-1;
    // openmp not support parallel for unordered_map
    std::vector<py::array_t<idtype>*> tmp_outs;
    for (size_t i = 0; i < outs.size(); i++){
        tmp_outs.push_back(&outs[i]);
        memset(outs[i].mutable_data(), 0, outs[i].size() * sizeof(idtype));// set zeros
    }
    
    if (!shape_match(data, treatment) || !shape_match(data, bin_x)){
        std::string msg = "The instance of `data`, `treatment` and `bin_x` doesn't match!"; 
        LightGBM::Log::Fatal(msg.c_str());
    }
    // checking out shape
    for (auto iter = outs.begin(); iter != outs.end(); ++iter){
        if (n_treatment < 0){
            n_treatment = iter->second.shape(0);
        }
        if (n_treatment != iter->second.shape(0)){
            LightGBM::Log::Fatal("the shape of outs[%d] doesn't match, n_treatment(%d!=%d)"
                , iter->first, iter->second.shape(0), n_treatment);
        }
        if (iter->first >= n_f){
            LightGBM::Log::Fatal("The key of `outs`(%d) out of range", iter->first);
        }
    }
    py::gil_scoped_release release;
    OMP_SET_NUM_THREADS(num_threads);
#pragma omp parallel for
    for (int f=0; f < tmp_outs.size(); ++f){// travel for all features
        auto &out = *tmp_outs[f];
        for (size_t i = 0; i < n_ins; i++){
            auto row = treatment.at(i, 0), col = bin_x.at(i, f);
            int cnt = 1;
            if (dropna == true){
                if(data.ndim() == 1 && std::isnan(data.at(i)) == true){
                    cnt = 0;
                }
                if(data.ndim() == 2){
                    cnt = 0;
                    for (size_t j = 0; j < n_t; j++){
                        if(std::isnan(data.at(i, j)) != true){
                            cnt = 1;
                            break;
                        }
                    }
                }
            }
            out.mutable_at(row, col) += cnt;
        }
    }
}

template <typename dtype, typename idtype>
py::array_t<dtype> & sumby_nogil(const py::array_t<dtype> &data,
                            const std::vector<py::array_t<idtype>> &by, 
                            py::array_t<dtype> &out, const std::string& transform){

    std::function<dtype(dtype)> transform_func;
    if("log" == transform){
        transform_func = [](dtype x){return std::log(x);};
    }else if ("square" == transform){
        transform_func = [](dtype x){return x*x;};
    }else if ("identity" == transform){
        transform_func = [] (dtype x){return x;};
    }else{
        throw std::runtime_error("Unknown transform function'"+transform+"'\n");
    }

    if (data.ndim() == 1){
        return sumby<dtype, idtype>(data, by[0], by[1], out, transform_func);
    }else{
        return sumby_2d<dtype, idtype>(data, by[0], by[1], out, transform_func);
    }
}


template <typename dtype, typename idtype>
py::array_t<idtype> & countby_nogil(const py::array_t<dtype> &data, 
                        std::vector<py::array_t<idtype>> &by, 
                        py::array_t<idtype> &out){
    return countby<dtype, idtype>(data, by, out, true);
}


template<typename dtype, typename idtype>
void indexbyarray(const py::array_t<dtype> & array, const py::array_t<idtype> & index, py::array_t<dtype> &out){
    auto shape = array.shape();
    auto uc_array = array.unchecked();
    auto uc_index = index.unchecked();
    auto uc_out = out.mutable_unchecked();

    size_t n = 1;
    for (size_t i = 1; i < out.ndim(); i++){
        n *= out.shape(i);
    }

    for (size_t i = 0; i < shape[0]; i++){
        auto pdest = static_cast<dtype*>(uc_out.mutable_data(i));
        auto psrc = static_cast<const dtype*>(uc_array.data(i, uc_index(i)));
        memcpy(pdest, psrc, sizeof(dtype) * n);        
    }
}

template<typename dtype, typename idtype>
void indexbyarray2(const py::array_t<dtype> & array, const py::array_t<idtype> & index, py::array_t<dtype> &out1, 
        py::array_t<dtype> &out2, int n_threads=-1){
    auto shape = array.shape();
    auto uc_array = array.unchecked();
    auto uc_index = index.unchecked();

    size_t n = 1;
    for (size_t i = 1; i < out1.ndim(); i++){
        n *= out1.shape(i);
    }
    if (n_threads < 0) {
      n_threads = OMP_NUM_THREADS();
    }
    py::gil_scoped_release release;
    omp_set_num_threads(n_threads);
#pragma omp parallel for
    for (size_t i = 0; i < shape[0]; i++){
        auto pdest = static_cast<dtype*>(out1.mutable_data(i));
        auto psrc = static_cast<const dtype*>(uc_array.data(i, uc_index(i)));
        memcpy(pdest, psrc, sizeof(dtype) * n);
        pdest = static_cast<dtype*>(out2.mutable_data(i));
        psrc = static_cast<const dtype*>(uc_array.data(i, 1 - uc_index(i)));
        memcpy(pdest, psrc, sizeof(dtype) * n);
    }
}

} // namespace gbct_utils
#endif //__AGGFN_HEADER__
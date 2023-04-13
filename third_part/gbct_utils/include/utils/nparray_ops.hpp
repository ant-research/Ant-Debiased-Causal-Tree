#ifndef __NPARRAY_OPS__
#define __NPARRAY_OPS__

#include <cmath>
#include <iostream>
#include <functional>
#include <memory>
#include <omp.h>
#include <unordered_map>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

namespace gbct_utils {


template <typename dtype>
void binary_op(const py::array_t<dtype> & a, const py::array_t<dtype> &b, py::array_t<dtype> &o, int a_offset, 
        int b_offset, int o_offset, int dim, std::function<dtype(dtype, dtype)> bin_func){
    auto pa = a.data() + a_offset, pb = b.data() + b_offset, po = o.mutable_data() + o_offset;
    if (dim + 1 == o.ndim()) { // last dimension
        if (a.shape(dim) == 1) {
            for (size_t i = 0; i < o.shape(dim); i++)
                po[i] = bin_func(pa[0], pb[i]);
        } else if (b.shape(dim) == 1) {
            for (size_t i = 0; i < o.shape(dim); i++)
                po[i] = bin_func(pa[i], pb[0]);
        } else {
            for (size_t i = 0; i < o.shape(dim); i++)
                po[i] = bin_func(pa[i], pb[i]);
        }
        return;
    }
    for (size_t i = 0; i < o.shape(dim); i++){
        binary_op(a, b, o, a_offset, b_offset, o_offset, dim + 1, bin_func);
        if (a.shape(dim) > 1){
            a_offset += a.strides(dim) / sizeof(dtype);
        }
        if (b.shape(dim) > 1){
            b_offset += b.strides(dim) / sizeof(dtype);
        }
        o_offset += o.strides(dim) / sizeof(dtype);
    }
}

template <typename dtype>
py::array_t<dtype> operator+(const py::array_t<dtype> &A,
                             const py::array_t<dtype> &B) {
    auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
    for (size_t i = 0; i < A.ndim(); i++){
        if(A.shape(i) != B.shape(i) && B.shape(i) > 1 && A.shape(i) > 1){
            LightGBM::Log::Fatal("all the input array dimensions for the concatenation axis must match exactly, "
            "but along dimension %d, the first array has size %d and the second array has size %d", 
            i, A.shape(i), A.shape(i));
        }
        shape[i] = std::max(A.shape(i), B.shape(i));
    }

    py::array_t<dtype> out(shape);
    std::function<dtype(dtype, dtype)> add_func = [](dtype a, dtype b){return a+b;};
    binary_op(A, B, out, 0, 0, 0, 0, add_func);
    return out;
};

template <typename dtype>
py::array_t<dtype> operator+(const py::array_t<dtype> &A, dtype B) {
  auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
  py::array_t<dtype> out(shape, A.data());
  auto ptr = out.mutable_data();
  for (size_t i = 0; i < A.size(); i++) {
    ptr[i] += B;
  }
  return out;
};

template <typename dtype>
py::array_t<dtype> operator+(dtype B, const py::array_t<dtype> &A) {
  return A + B;
};

template <typename dtype>
py::array_t<dtype> operator-(const py::array_t<dtype> &A,
                             const py::array_t<dtype> &B) {
    auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
    for (size_t i = 0; i < A.ndim(); i++){
        if(A.shape(i) != B.shape(i) && B.shape(i) > 1 && A.shape(i) > 1){
            LightGBM::Log::Fatal("all the input array dimensions for the concatenation axis must match exactly, "
            "but along dimension %d, the first array has size %d and the second array has size %d", 
            i, A.shape(i), A.shape(i));
        }
        shape[i] = std::max(A.shape(i), B.shape(i));
    }

    py::array_t<dtype> out(shape);
    std::function<dtype(dtype, dtype)> minus_func = [](dtype a, dtype b) {return a - b;};
    binary_op(A, B, out, 0, 0, 0, 0, minus_func);
    return out;
};

template <typename dtype>
py::array_t<dtype> operator-(const py::array_t<dtype> &A, dtype B) {
  auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
  py::array_t<dtype> out(shape, A.data());
  auto ptr = out.mutable_data();
  for (size_t i = 0; i < A.size(); i++) {
    ptr[i] -= B;
  }
  return out;
};

template <typename dtype>
py::array_t<dtype> operator-(dtype B, const py::array_t<dtype> &A) {
  auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
  py::array_t<dtype> out(shape, A.data());
  auto ptr = out.mutable_data();
  for (size_t i = 0; i < A.size(); i++) {
    ptr[i] = B - ptr[i];
  }
  return out;
};

template <typename dtype>
py::array_t<dtype> operator-(const py::array_t<dtype> &A) {
  auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
  py::array_t<dtype> out(shape);
  auto ptr = out.mutable_data();
  auto pdata = A.data();
  for (size_t i = 0; i < A.size(); i++) {
    ptr[i] = - pdata[i];
  }
  return out;
};

template <typename dtype>
py::array_t<dtype> operator/(const py::array_t<dtype> &A, dtype B) {
  auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
  py::array_t<dtype> out(shape, A.data());
  auto ptr = out.mutable_data();
  for (size_t i = 0; i < A.size(); i++) {
    ptr[i] /= B;
  }
  return out;
};

template <typename dtype>
py::array_t<dtype> operator/(const py::array_t<dtype> &A,
                             const py::array_t<dtype> &B) {
    auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
    for (size_t i = 0; i < A.ndim(); i++){
        if(A.shape(i) != B.shape(i) && B.shape(i) > 1 && A.shape(i) > 1){
            LightGBM::Log::Fatal("all the input array dimensions for the concatenation axis must match exactly, "
            "but along dimension %d, the first array has size %d and the second array has size %d", 
            i, A.shape(i), A.shape(i));
        }
        shape[i] = std::max(A.shape(i), B.shape(i));
    }

    py::array_t<dtype> out(shape);
    std::function<dtype(dtype, dtype)> div_func = [](dtype a, dtype b) {return a / b;};
    binary_op(A, B, out, 0, 0, 0, 0, div_func);
    return out;
};

template <typename dtype>
py::array_t<dtype> operator/(dtype B, const py::array_t<dtype> &A) {
  auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
  py::array_t<dtype> out(shape, A.data());
  auto ptr = out.mutable_data();
  for (size_t i = 0; i < A.size(); i++) {
    ptr[i] = B / ptr[i];
  }
  return out;
};

template <typename dtype>
py::array_t<dtype> operator*(const py::array_t<dtype> &A,
                             const py::array_t<dtype> &B) {
    auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
    for (size_t i = 0; i < A.ndim(); i++){
        if(A.shape(i) != B.shape(i) && B.shape(i) > 1 && A.shape(i) > 1){
            LightGBM::Log::Fatal("all the input array dimensions for the concatenation axis must match exactly, "
            "but along dimension %d, the first array has size %d and the second array has size %d", 
            i, A.shape(i), A.shape(i));
        }
        shape[i] = std::max(A.shape(i), B.shape(i));
    }

    py::array_t<dtype> out(shape);
    std::function<dtype(dtype, dtype)> mul_func = [](dtype a, dtype b) {return a * b;};
    binary_op(A, B, out, 0, 0, 0, 0, mul_func);
    return out;
};

template <typename dtype>
py::array_t<dtype> operator*(const py::array_t<dtype> &A, dtype B) {
  auto shape = std::vector<ssize_t>(A.shape(), A.shape() + A.ndim());
  py::array_t<dtype> out(shape, A.data());
  auto ptr = out.mutable_data();
  for (size_t i = 0; i < A.size(); i++) {
    ptr[i] *= B;
  }
  return out;
};

template <typename dtype>
py::array_t<dtype> operator*(dtype B, const py::array_t<dtype> &A) {
  return A * B;
};

// sum
template <typename dtype>
dtype sum(const py::array_t<dtype> &m) {
    dtype out=0;
    auto pdata = m.data();
    for (size_t i = 0; i < m.size(); i++){
        out += pdata[i];
    }
    return out;
}
template <typename dtype>
py::array_t<dtype> sum(const py::array_t<dtype> &m, int axis, bool keep_dims=false) {
    if(m.ndim() <= axis){
        LightGBM::Log::Fatal("axis %d is out of bounds for array of dimension %d", axis, m.ndim());
    }
    std::vector<ssize_t> out_shape(m.shape(), m.shape()+m.ndim());
    out_shape[axis] = 1;
    //create a out array
    py::array_t<dtype> out(out_shape);
    dtype* pout = static_cast<dtype*>(out.mutable_data());
    memset(pout, dtype(0), out.size() * sizeof(dtype));

    const dtype *pdata = static_cast<const dtype*>(m.data());
    auto up_stride = 0, low_stride = out.strides(axis)/sizeof(dtype);
    if(axis>0){
      up_stride = out.strides(axis-1)/sizeof(dtype);
    }

    for (size_t i = 0; i < m.size(); i++){
        auto i_ = i - (i/low_stride)*low_stride + int(i/(up_stride+0.0001))*up_stride;
        pout[i_] += pdata[i];
    }
    if (keep_dims == false){
      out_shape.erase(out_shape.begin() + axis);
    }
    
    return out.reshape(out_shape);
}

template <typename dtype>
void *_concatenate(const py::array_t<dtype> &A, const py::array_t<dtype> &B, void *out, int A_off, int B_off, 
        int cur_axis, int axis) {
    if (cur_axis == axis){
        auto nbytes = A.strides(axis) * A.shape(axis);
        memcpy(out, A.data() + A_off, nbytes);
        memcpy(out + nbytes, B.data() + B_off, B.strides(axis) * B.shape(axis));
        return out + B.strides(axis) * (B.shape(axis) + A.shape(axis));
    }else{
        for (size_t i = 0; i < A.shape(cur_axis); i++){
            out = _concatenate(A, B, out, A_off, B_off, cur_axis+1, axis);
            A_off += A.strides(cur_axis)/sizeof(dtype);
            B_off += B.strides(cur_axis)/sizeof(dtype);
        }
    }
    return out;
}

template <typename dtype>
py::array_t<dtype> concatenate(const py::array_t<dtype> &A, const py::array_t<dtype> &B, int axis){
    if(A.ndim() != B.ndim()){
      LightGBM::Log::Fatal("all the input array dimensions must be the same length.");
    }
    std::vector<ssize_t> out_shape(A.shape(), A.shape()+A.ndim());
    shapes_match(A, B);
    out_shape[axis] += B.shape(axis);

    py::array_t<dtype> out(out_shape);
    auto stride = out.strides(axis)/sizeof(dtype);
    auto pout = out.mutable_data();

    _concatenate(A, B, pout, 0, 0, 0, axis);
    return out;
}

} // namespace gbct_utils

#endif //__NPARRAY_OPS__
#!/usr/bin/env python
# -*-coding:utf-8 -*-

from typing import List
import logging
import numpy as np

from gbct_utils import common, bin


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


countby_fn_map = {
    'float64_int64': common.countby_double_int64,
    'double_int64': common.countby_double_int64,
    'float64_uint64': common.countby_double_uint64,
    'double_uint64': common.countby_double_uint64,
    'float32_int64': common.countby_float_int64,
    'float32_uint64': common.countby_float_uint64,
    'float64_int32': common.countby_double_int32,
    'double_int32': common.countby_double_int32,
    'float64_uint32': common.countby_double_uint32,
    'double_uint32': common.countby_double_uint32,
    'float32_int32': common.countby_float_int32,
    'float32_uint32': common.countby_float_uint32,
    'int32_int32': common.countby_int32_int32,
    'int64_int32': common.countby_int64_int32,
}

sumby_fn_map = {
    'double_int32': common.sumby_double_int32,
    'float64_int32': common.sumby_double_int32,
    'float32_uint32': common.sumby_float_uint32,
    'float32_int32': common.sumby_float_int32,
    'float32_int64': common.sumby_float_int64,
    'int32_int32': common.sumby_int32_int32,
    'int64_int32': common.sumby_int64_int32,
    'int64_int64': common.sumby_int64_int64,
}

indexbyarray_fn_map = {
    'double_int32': common.indexbyarray_double_int32,
    'double_uint32': common.indexbyarray_double_uint32,
    'float64_int32': common.indexbyarray_double_int32,
    'float64_uint32': common.indexbyarray_double_uint32,
    'float32_int32': common.indexbyarray_float_int32,
    'float32_uint32': common.indexbyarray_float_uint32,
    'float32_int64': common.indexbyarray_float_int64,
    'float32_uint64': common.indexbyarray_float_uint64
}

update_histogram_fn_map = {
    'update_histogram_int32_int32': common.update_histogram_int32_int32,
    'update_histogram_double_int32': common.update_histogram_double_int32,
    'update_histogram_float64_int32': common.update_histogram_double_int32
}

Value2BinParallel_fn_map = {
    'double_int32': bin.Value2BinParallel_double_int32,
    'double_uint32': bin.Value2BinParallel_double_uint32,
    'float64_int32': bin.Value2BinParallel_double_int32,
    'float64_uint32': bin.Value2BinParallel_double_uint32,
    'float32_int32': bin.Value2BinParallel_float_int32,
    'float32_uint32': bin.Value2BinParallel_float_uint32
}

def _check_match(*args, axis=0):
    m = len(args)
    n = None
    for arg in args:
        assert len(arg.shape) > axis, f'The given axis({axis}) out of range!'
        if n is None:
            n = arg.shape[axis]
        assert n == arg.shape[axis], f'The shape on dimension of axis={axis} doesn\'t match!'


def _check_c_style_array(*args):
    for array in args:
        if isinstance(array, np.ndarray):
            assert array.flags.c_contiguous, f'input array must be c_style stored!'


def countby(data, by, out):
    """Counting the rows in the data by the values in the `by`. its function is similar to `groupby.count` in pandas.

    Args:
        data (_type_): _description_
        by (_type_): conditions of counby, dtype must be int32 or int64
        out (_type_): the result of countby

    Note: it is implemented in C++, and its core code removes the Gil restriction. 
    Therefore, multi-core computing can be realized directly with Python multithreading.

    Returns:
        _type_: the result of countby. Notice: it shares the same memory with parameter `out`.
    """
    # dtype check
    dtype = data.dtype
    itype = out.dtype
    if len(by) not in (1, 2):
        raise ValueError(f'The dimension of by must be 1 or 2, but got {len(by)}')
    fn_key = f'{dtype.name}_{itype.name}'
    if fn_key not in countby_fn_map:
        logger.fatal(f'{fn_key} not supported')
        raise ValueError(f'{fn_key} not supported')
        # GlobalMatrix type transform
    _check_c_style_array(data, *by, out)
    fn = countby_fn_map[fn_key]
    return fn(data, by, out)


def sumby(data, by, out, transform="identity"):
    """Summing by the values in the `by`. its function is similar to `groupby.sum` in pandas.
    
    Note: it is implemented in C++, and its core code removes the Gil restriction. 
    Therefore, multi-core computing can be realized directly with Python multithreading.

    Args:
        data (_type_): _description_
        by (_type_): conditions of sumby, dtype must be int32 or int64. Notices: less than 2 dimensions are supported.
        out (_type_): _description_

    Returns:
        _type_: the result of countby. Notice: it shares the same memory with parameter `out`.
    """
    dtype = data.dtype
    if len(by) not in (1, 2):
        raise ValueError(f'The dimension of by must be 1 or 2, but got {len(by)}')
    itype = by[0].dtype
    fn_key = f'{dtype.name}_{itype.name}'
    if fn_key not in sumby_fn_map:
        logger.fatal(f'{fn_key} not supported')
        raise ValueError(f'{fn_key} not supported')
    if len(data.shape) >= 3:
        raise ValueError(f'The dimension of data must be less than 3')
    if len(data.shape) == 2 and data.shape[1] != out.shape[-1]:
        raise ValueError(f'The shape of data({data.shape}) don\'t match the shape of out({out.shape})')
    fn = sumby_fn_map[fn_key]
    return fn(data, by, out, transform)


def indexbyarray(arr, idx, out):
    assert arr.dtype == out.dtype, f'arr.dtype({arr.dtype}) != out.dtype({out.dtype})!'
    assert arr.shape[0] == idx.shape[0], f'arr.shape({arr.shape}) != idx.shape({idx.shape})!'
    assert arr.shape[0] == out.shape[0], f'arr.shape({arr.shape}) != out.shape({out.shape})!'
    _check_c_style_array(arr, idx, out)
    fn = indexbyarray_fn_map[f'{arr.dtype.name}_{idx.dtype.name}']
    return fn(arr, idx, out)


def update_x_map(x_binned, ins2leaf, split_infos, leaves_range, out):
    """Update the x_binned map with the new binned data.
    x_binned: [n, nfeature], 特征
    ins2leaf: [n, ], 样本到叶子的映射
    split_infos: [n_leaf, 2], 分裂信息 <feature_id, feature_value>
    leaves_range: [n_leaf, 2], 叶子对应的数据范围
    out: [n_leaf*2, 2], 输出
    """
    assert x_binned.shape[0] == ins2leaf.shape[0] and split_infos.shape[0] == leaves_range.shape[0]
    dtype = x_binned.dtype
    assert dtype == ins2leaf.dtype and dtype == split_infos.dtype and dtype == leaves_range.dtype and dtype == out.dtype
    _check_c_style_array(x_binned, ins2leaf, out)
    common.update_x_map_int32(x_binned, ins2leaf, split_infos, leaves_range, out)


def update_histogram(target, x_binned, index, leaves_range, treatment, out, n_treatment=2, n_bins=64, threads=20):
    """update the histogram of each leaf.

    Args:
        target (_type_): [n, n_outcome]
        x_binned (_type_): [n, n_feature]
        index (_type_): [n_], end_pos in leaves_range must be no greater than n_
        leaves_range (_type_): [n_leaf, 2], list of each leaf's data range, each term looks like [st_pos, end_pos).
        treatment (_type_): [n]
        out (_type_): [n_leaf, n_features, n_bins, n_treatment, n_outcome]. The output histogram.
        n_treatment (int, optional): The number of treatment. Defaults to 2.
        n_bins (int, optional): _description_. Defaults to 64.
        threads (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: the same with `out`
    """
    dtype = target.dtype
    itype = x_binned.dtype
    assert dtype == out.dtype, f'the `target`({dtype}) must be the same with `out`({out.dtype})'
    assert (itype == x_binned.dtype and itype == index.dtype and itype == leaves_range.dtype
            and itype == treatment.dtype), f'except `target` and `out`, other parameter must be the same dtype!'
    n_outcome = target.shape[1]
    n_feature = x_binned.shape[1]
    n_leaf = len(leaves_range)
    n_i = index.shape[0]
    _check_match(target, x_binned, treatment)
    _check_c_style_array(target, x_binned, index, leaves_range, treatment, out)
    for _, end in leaves_range:
        assert end <= n_i, f'leaf range out of `index`'
    # out shape check
    assert (out.shape == (n_leaf, n_feature, n_bins, n_treatment, n_outcome),
            f'the shape of `out` ({out.shape}) not equals to {[n_leaf, n_feature, n_bins, n_treatment, n_outcome]}!')
    fn_key = f'update_histogram_{dtype.name}_{itype.name}'
    fn = update_histogram_fn_map[fn_key]
    return fn(target, x_binned, index, leaves_range, treatment, out, n_treatment, n_bins, threads)


def FindBinParallel(data,
            max_bin=64,
            min_data_in_bin=100,
            min_split_data=100,
            pre_filter=False,
            bin_type=0,
            use_missing=True,
            zero_as_missing=False,
            forced_upper_bounds=[]):
    dtype = data.dtype
    _check_c_style_array(data)
    if dtype.name in ('double', 'float64'):
        return bin.FindBinParallel_double(data, max_bin, min_data_in_bin, min_split_data, pre_filter, bin_type, use_missing,
                        zero_as_missing, forced_upper_bounds)
    else:
        raise ValueError(f'The {dtype.name} has not been supported!')


def Value2BinParallel(data, bin_mappers: List[bin.BinMaper], out=None, threads=-1):
    if out is None:
        out = np.zeros_like(data, np.int32)
    assert out.dtype in (np.int32, np.uint32), f'out dtype must be {np.int32} or {np.uint32}!'
    _check_c_style_array(data, out)
    dtype = data.dtype
    idtype = out.dtype
    fn_key = f'{dtype.name}_{idtype.name}'
    fn = Value2BinParallel_fn_map[fn_key]

    assert isinstance(data, np.ndarray), f'Only np.ndarray is supported!'
    return fn(data, bin_mappers, out, threads)

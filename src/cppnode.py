# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
from typing import List

from gbct_utils import common as gbct_utils

indexbyarray_fn_map = {
    'double_int32': gbct_utils.indexbyarray2_double_int32,
    'double_uint32': gbct_utils.indexbyarray2_double_uint32,
    'float64_int32': gbct_utils.indexbyarray2_double_int32,
    'float64_uint32': gbct_utils.indexbyarray2_double_uint32,
    'float32_int32': gbct_utils.indexbyarray2_float_int32,
    'float32_uint32': gbct_utils.indexbyarray2_float_uint32,
    'float32_int64': gbct_utils.indexbyarray2_float_int64,
    'float32_uint64': gbct_utils.indexbyarray2_float_uint64
}

def create_didnode_from_dict(info):
    # basic information for tree node
    # basic_keys = ['children', 'split_feature', 'split_thresh', 'is_leaf', 'leaf_id', 'level_id']
    assert len(info['children']) == 2, f'children should be 2!'
    node = gbct_utils.CppDebiasNode()
    basic_keys = node.get_property_keys()
    for k in info.keys():
        if k in basic_keys:
            setattr(node, k, info[k])
        else:
            node.set_info(k, info[k])
    return node


def predict(nodes:List[gbct_utils.CppDiDNode], x, out, key, threads=20, batch_size=1024):
    if len(nodes) <= 0:
        raise RuntimeError(f'The number of nodes must be greater than 0!')
    elif isinstance(nodes[0], list) is False:
        nodes = [nodes]

    if isinstance(nodes[0][0], gbct_utils.CppDiDNode):
        return gbct_utils.predict_did(nodes, out, x, key, threads)
    elif isinstance(nodes[0][0], gbct_utils.CppDebiasNode):
        return gbct_utils.predict_debias(nodes, out, x, key, threads)
    else:
        raise ValueError(f'{type(nodes[0][0])} is not supported!')


def indexbyarray(arr, idx, fact_outcome, counterfact_outcome, n_threads=-1):
    assert arr.dtype == fact_outcome.dtype, f'arr.dtype({arr.dtype}) != fact_outcome.dtype({fact_outcome.dtype})!'
    assert arr.shape[0] == idx.shape[0], f'arr.shape({arr.shape}) != idx.shape({idx.shape})!'
    assert arr.shape[0] == fact_outcome.shape[0], f'arr.shape({arr.shape}) != out.shape({fact_outcome.shape})!'
    assert (fact_outcome.shape == counterfact_outcome.shape and counterfact_outcome.dtype
            == fact_outcome.dtype), f'fact_outcome and counterfact_outcome should be the same shape and dtype!'
    fn = indexbyarray_fn_map[f'{arr.dtype.name}_{idx.dtype.name}']
    return fn(arr, idx, fact_outcome, counterfact_outcome, n_threads)

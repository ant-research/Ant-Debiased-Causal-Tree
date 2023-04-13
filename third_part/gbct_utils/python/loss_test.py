#!/usr/bin/env python
# -*-coding:utf-8 -*-

from time import time
import unittest as ut

import numpy as np
from gbct_utils import common


class TestLoss(ut.TestCase):
    def test_sum(self):
        shape = [1000, 2]
        data = np.random.normal(0, 1, size=shape)
        result = common.sum(data, 0, False)
        self.assertTrue(np.array_equal(result, data.sum(0)))
        result = common.sum(data, 0, True)
        self.assertTrue(np.array_equal(result, data.sum(0, keepdims=True)))

    def test_concatenate(self):
        for _ in range(10):
            ndim = np.random.randint(1, 5)
            shape = np.random.randint(1, 5, size=[ndim])
            data = np.random.normal(0, 1, size=shape)
            axis = np.random.randint(0, ndim)
            result = common.concatenate(data, data, axis)
            self.assertTrue(np.array_equal(result, np.concatenate([data, data], axis)))

    def test_calculate_loss(self):
        n_treats, n_bins, n_outs, n_feats, n_leafs = 2, 2, 8, 2, 1
        # [n_leaf, n_features, n_bins, n_treatment, n_outcome]
        bin_grad_hist = np.random.normal(0, 1, size=[n_leafs, n_feats, n_bins, n_treats, n_outs])
        bin_hess_hist = np.random.normal(0, 1, size=[n_leafs, n_feats, n_bins, n_treats, n_outs])
        bin_cgrad_hist = np.random.normal(0, 1, size=[n_leafs, n_feats, n_bins, n_treats, n_outs])
        bin_chess_hist = np.random.normal(0, 1, size=[n_leafs, n_feats, n_bins, n_treats, n_outs])
        bin_counts = np.random.randint(10, 100, size=[n_leafs, 1, n_bins, n_treats])
        lambd, coeff, t0 = 5.0, 1.0, 7
        configs={0:{0:[0,1,2,3]}}
        common.calculate_loss(configs, bin_grad_hist, bin_cgrad_hist, bin_hess_hist, bin_chess_hist, bin_counts, lambd,
                              coeff, t0)

    def test_array_add(self):
        a = np.random.normal(0, 1, size=[10, 1, 10, 1])
        b = np.random.normal(0, 1, size=[10, 2, 1, 2])
        self.assertTrue(np.array_equal(common.array_add(a, b), a + b))


if __name__ == '__main__':
    ut.main()

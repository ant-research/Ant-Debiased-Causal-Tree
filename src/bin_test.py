# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

import pickle
import unittest as ut

import numpy as np
from pyhocon import ConfigFactory

from bin import BinMapper


class TestBin(ut.TestCase):

    def setUp(self):
        self.conf = ConfigFactory.parse_file('config/gbct.conf')

    def test_find_bin(self):
        n, m = 100000, 2
        x = np.random.normal(0, 1, size=(n, m))
        b = BinMapper(self.conf)
        b.fit(x)
        bin_x = b.transform(x)
        for i in range(m):
            values, cnt = np.unique(bin_x[:, i], return_counts=True)
            cumcnt = 0
            for j, _cnt in zip(values, cnt):
                cumcnt += _cnt
                self.assertEqual((x[:, i] <= b.upper_bounds[i, j]).sum(), cumcnt)

    def test_pickle_binmaper(self):
        n, m = 100000, 2
        x = np.random.normal(0, 1, size=(n, m))
        b = BinMapper(self.conf)
        b.fit(x)
        path = "binmaper.pkl"
        with open(path, "wb") as f:
            pickle.dump(b, f)
        with open(path, "rb") as f:
            _b = pickle.load(f)
        self.assertTrue(np.array_equal(b.upper_bounds, _b.upper_bounds))


if __name__ == '__main__':
    np.random.seed(9)
    ut.main()
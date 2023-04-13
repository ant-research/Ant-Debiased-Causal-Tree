#!/usr/bin/env python
# -*-coding:utf-8 -*-

from time import time
import unittest as ut
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from aggrate import *


class TestAggrate(ut.TestCase):

    def test_sumby(self):
        n = 100
        pool = ThreadPoolExecutor(10)
        data = np.arange(2 * n, dtype=np.float64).reshape([n, 2])
        by = [
            np.random.randint(0, 100, size=(n, ), dtype=np.int32),
            np.random.randint(0, 100, size=(n, ), dtype=np.int32)
        ]
        future_list = []
        result = []
        st = time()
        for i in range(12):
            result.append(np.zeros([100, 100, 2], dtype=data.dtype))
            future = pool.submit(sumby, data, by, result[-1])
            future_list.append(future)
        pool.shutdown(wait=True)
        print(f'sumby: {time() - st:.3f}s')
        # for res in result:
        for i, future in enumerate(futures.as_completed(future_list)):
            res = future.result()
            self.assertEqual(res.sum(), data.sum())

    def test_sumby_parallel(self):
        n = 10000
        n_f = 32
        n_bin = 64
        n_treatment = 2
        n_t = 1
        data = np.ones([n_t * n], dtype=np.float64).reshape([n, n_t])
        by = [
            np.random.randint(0, n_treatment, size=(n, 1), dtype=np.int32),
            np.random.randint(0, n_bin, size=(n, n_f), dtype=np.int32)
        ]
        st = time()
        result = {f: np.zeros([n_treatment, n_bin, n_t], dtype=data.dtype) for f in range(n_f)}
        sumby_parallel(data, by, result)
        print(f'test_sumby_parallel: {time() - st:.3f}s')
        # for res in result:
        for k, v in result.items():
            target = pd.DataFrame(np.concatenate([data, by[0], by[1][:, k:k + 1]], -1)).groupby([1, 2]).sum()
            for i in range(n_treatment):
                for j in range(n_bin):
                    self.assertEqual(v[i, j, 0], target.loc[i, j][0])

    def test_countby_parallel(self):
        n = 10000
        n_f = 32
        n_bin = 64
        n_treatment = 2
        n_t = 2
        data = np.ones([n_t * n], dtype=np.float64).reshape([n, n_t])
        by = [
            np.random.randint(0, n_treatment, size=(n, 1), dtype=np.int32),
            np.random.randint(0, n_bin, size=(n, n_f), dtype=np.int32)
        ]
        st = time()
        result = {f: np.zeros([n_treatment, n_bin], dtype=np.int32) for f in range(n_f)}
        countby_parallel(data, by, result)
        print(f'test_countby_parallel: {time() - st:.3f}s')
        # for res in result:
        for k, v in result.items():
            target = pd.DataFrame(np.concatenate([data, by[0], by[1][:, k:k + 1]], -1)).groupby([n_t, n_t + 1]).count()
            for i in range(n_treatment):
                for j in range(n_bin):
                    self.assertEqual(v[i, j], target.loc[i, j][0])

    def test_countby(self):
        n = 1000
        n_bys = 2
        pool = ThreadPoolExecutor()
        data = np.arange(2 * n, dtype=np.float64).reshape([n, 2])
        by = [np.random.randint(0, 100, size=(n, ), dtype=np.int32) for _ in range(n_bys)]
        future_list = []
        result = []
        for i in range(12):
            result.append(np.zeros([100] * n_bys, dtype=np.int32))
            future = pool.submit(countby, data, by, result[-1])
            future_list.append(future)

        pool.shutdown(wait=True)
        for i, future in enumerate(futures.as_completed(future_list)):
            self.assertEqual(result[i].sum(), data.shape[0])

        # test nan
        pool = ThreadPoolExecutor()
        n = 1000
        data = np.arange(2 * n, dtype=np.float64).reshape([n, 2])
        by = [
            np.random.randint(0, 100, size=(n, ), dtype=np.int32),
            np.random.randint(0, 100, size=(n, ), dtype=np.int32)
        ]
        data[data[:, 0] % 3 == 0, :] = np.nan
        future_list = []
        result = []
        for i in range(5):
            result.append(np.zeros([100, 100], dtype=np.int32))
            future = pool.submit(countby, data, by, result[-1])
            future_list.append(future)
        pool.shutdown(wait=True)
        for i, future in enumerate(futures.as_completed(future_list)):
            self.assertEqual(result[i].sum(), n - np.isnan(data[:, 0]).sum())

    def test_update_x_map(self):
        n, m, l, n_y, n_bins = 100, 10, 10, 2, 10
        x = np.random.randint(0, n_bins, size=(n, m), dtype=np.int32)
        index = np.random.permutation(n).astype(np.int32)
        split_infos = np.concatenate([
            np.random.randint(0, m, size=(l, 1), dtype=np.int32),
            np.random.randint(0, n_bins - 1, size=(l, 1), dtype=np.int32)
        ],
                                     axis=1)
        tmp = np.sort(np.random.choice(range(n), size=(l+1, ), replace=False))
        leaves_range = np.stack([tmp[:-1], tmp[1:]], axis=1).astype(np.int32)
        print(leaves_range)
        out = np.zeros([l * 2, 2], np.int32)
        update_x_map(x, index, split_infos, leaves_range, out)
        # check split
        for i in range(l):
            self.assertEqual(out[i * 2, 0], leaves_range[i, 0])
            self.assertEqual(out[i * 2 + 1, 1], leaves_range[i, 1])
        for i in range(l):
            fid, thresh = split_infos[i]
            fr, end = out[i * 2]
            for j in range(fr, end):
                self.assertLessEqual(x[index[j], fid], thresh)
            fr, end = out[i * 2 + 1]
            for j in range(fr, end):
                self.assertGreater(x[index[j], fid], thresh)

    def test_update_histogram(self):
        n, m, l, n_y, n_w, n_bins = 100, 100, 10, 2, 2, 10
        x = np.random.randint(0, n_bins, size=(n, m), dtype=np.int32)
        y = np.random.normal(size=(n, n_y))
        w = np.random.randint(0, n_w, size=(n, ), dtype=np.int32)
        index = np.random.permutation(n).astype(np.int32)
        tmp = [0] + list(np.sort(np.random.choice(range(n), size=(l - 1, ), replace=False))) + [n]
        leaves_range = np.stack([tmp[:-1], tmp[1:]], axis=1).astype(np.int32)
        out = np.zeros([l, m, n_bins, n_w, n_y], y.dtype)
        print('*' * 50)
        time_start = time()
        update_histogram(y, x, index, leaves_range, w, out, n_w, n_bins, 32)
        print(f'elapsed time: {time() - time_start:.3f}s')
        print('*' * 50)
        df_y = pd.DataFrame(y)
        for f in range(m):
            for leaf in range(l):
                fr, end = leaves_range[leaf]
                idx = index[fr:end]
                tmp = df_y.loc[idx].groupby([x[idx, f], w[idx]]).sum()
                for i in range(n_bins):
                    for j in range(n_w):
                        if (i, j) in tmp.index:
                            self.assertTrue(np.abs(out[leaf, f, i, j] - tmp.loc[i, j]).mean() < 1e-6)
                        else:
                            self.assertEqual(np.abs(out[leaf, f, i, j]).sum(), 0)

    def test_find_bin(self):
        n, m, n_bins = 10000, 100, 64
        x = np.random.normal(0, 3, size=(n, m))
        bin_mappers = FindBinParallel(x, n_bins)
        print(Value2BinParallel(x, bin_mappers))


    def test_indexbyarray2(self):
        n = 10000
        n_treatment = 2
        n_outcome = 8
        y = np.random.normal(0, 1, size=(n, n_treatment, n_outcome))
        by = np.random.randint(0, n_treatment, size=(n), dtype=np.int32)
        o1 = np.zeros([n, n_outcome], dtype=y.dtype)
        o2 = np.zeros([n, n_outcome], dtype=y.dtype)
        indexbyarray2(y, by, o1, o2)
        for i in range(n):
            self.assertTrue(np.array_equal(y[i, by[i]], o1[i]))
            self.assertTrue(np.array_equal(y[i, 1-by[i]], o2[i]))


if __name__ == '__main__':
    ut.main()

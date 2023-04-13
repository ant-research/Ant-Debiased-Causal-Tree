#!/usr/bin/env python
# coding: utf-8

"""
@Author  :   C.Z. Tang
"""

import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from pyhocon import ConfigFactory

seed = 9  # set random seed
np.random.seed(seed)
random.seed(seed)


def train_test(path:str, cv=2, n=10, bias=True):
    conf_path = os.path.join(path, 'synthetic.conf')
    conf = ConfigFactory.parse_file(conf_path)
    filenames = conf.data.path
    path_prefix = path # os.path.dirname(filenames[1])
    tr_unit = pd.read_csv(os.path.join(path, os.path.basename(filenames[0])), header=0, index_col=0)
    tr_series = pd.read_csv(os.path.join(path, os.path.basename(filenames[1])), header=[0, 1], index_col=0)
    te_unit = pd.read_csv(os.path.join(path_prefix,f'test_{os.path.basename(filenames[0])}'), header=0, index_col=0)
    te_series = pd.read_csv(os.path.join(path_prefix,f'test_{os.path.basename(filenames[1])}'), header=[0, 1], index_col=0)
    series = tr_series.append(te_series)
    unit = tr_unit.append(te_unit)
    # randomly split
    tr_ratio = (cv-1)/cv
    indexs = []
    if bias is True:
        mean = series['y0'][series['y0'].columns[:-1]].mean().mean()
        temp = series['y0'][series['y0'].columns[:-1]].mean(1)
        bias_idx = ((temp < mean ) & (unit['treatment']==0))|((temp > mean  ) & (unit['treatment']==1))
    all_index = np.arange(series.shape[0])
    for i in range(n):
        ind = train_test_split(all_index, train_size=tr_ratio, random_state=seed+i)
        # sort for idex
        if bias:
            ind[0] = ind[0][bias_idx[ind[0]]]
        
        ind = [np.sort(idx) for idx in ind]
        indexs.append(ind)
    indexs = np.array(indexs)
    # export
    np.save(os.path.join(path_prefix, 'tr_te.npy'), indexs)
    series.to_csv(os.path.join(path_prefix, 'Y.csv'))
    unit.to_csv(os.path.join(path_prefix, 'X.csv'))


if __name__ == '__main__':
    train_test('data/binary_var_1_y')
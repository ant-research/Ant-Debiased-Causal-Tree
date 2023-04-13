# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

import os
import pickle

import numpy as np
import pandas as pd
from pyhocon import ConfigTree, ConfigFactory

from dataset import Dataset
from aggregate import countby, sumby, update_histogram
from information import GBCTDataInfo


class Sim2Dataset(Dataset):
    def __init__(self, conf: ConfigTree):
        super(Sim2Dataset, self).__init__()
        self.conf = conf
        self.info = GBCTDataInfo(conf=conf)
        self.pretreat_mask = []
        self.posttreat_mask = []
        self.bin_features = None
        data_conf = conf.get('dataset', conf)
        treat_dt = data_conf.get_int('$.treat_dt')
        periods = data_conf.get_list('$.periods')
        target_names = data_conf.get_list('$.target')
        for i in periods:
            for name in target_names:
                if i < treat_dt:
                    self.pretreat_mask.append((name, i))
                else:
                    self.posttreat_mask.append((name, i))

    def read(self, filenames):
        print(filenames)
        conf = self.conf.dataset
        if len(filenames) == 2:
            unit = pd.read_csv(filenames[0], header=0, index_col=0)
            series = pd.read_csv(filenames[1], header=[0, 1], index_col=0)
            series.columns = pd.MultiIndex.from_tuples([(i, int(j)) for i, j in series.columns])
            # 读取simulation conf
            path = os.path.join(os.path.dirname(filenames[0]), 'simulation.conf')
            if os.path.exists(path):
                conf.put('simulation', ConfigFactory.parse_file(path))
        else:
            temp = pickle.load(open(filenames, 'rb'))
            unit = temp['unit']
            series = temp['series']
        temp, drop_cols = [], []
        for c in unit.columns:
            if c == 'treatment':
                temp.append(('treatment', c))
            elif c in conf.feature:
                temp.append(('features', c))
            else:
                drop_cols.append(c)
        unit = unit.drop(columns=drop_cols)
        unit.columns = pd.MultiIndex.from_tuples(temp)
        self._data = pd.concat([unit, series], axis=1)

        self.treatment = self._data[('treatment', 'treatment')].astype(np.int32)
        self.treatment.name = 'treatment'
        self.n_period = self.targets.shape[1]
        self.n_inst = self.targets.shape[0]
        self.n_feat = self.features.shape[1]

    def get_pretreated_targets(self):
        return self.targets[self.targets.columns[:-self.info.treat_dt]]

    def sub_dataset(self, index=None, cols=None, cols_y=None) -> Dataset:
        if index.dtype in (pd.BooleanDtype, np.bool):
            assert index.shape[0] == self.n_inst
            index = np.where(index)[0]
        data = Sim2Dataset(self.conf)
        data.n_inst = index.shape[0]
        if cols is None:
            data.n_feat = self.n_feat
            cols = self.features.columns
        else:
            data.n_feat = len(cols)

        if cols_y is None:
            cols_y = self.targets.columns

        all_cols = []
        for c in self._data.columns:
            if c[0] == 'features' and c[1] not in cols:
                continue
            if c[0] == 'y0' and ('y', c[1]) in cols_y:
                continue
            if c[0] == 'y' and c not in cols_y:
                continue
            all_cols.append(c)

        data.n_period = self.n_period
        data.n_feat = len(cols) + len(self.targets.columns) - len(cols_y)
        data._data = self._data.iloc[index][all_cols]
        data.treatment = self.treatment.iloc[index]
        return data

    @property
    def targets(self):
        return self._data[[('y', i) for i in self._data['y'].columns]]

    @property
    def fact_outcome(self):
        return self.targets

    @property
    def counterfact_outcome(self, w: int = 0):
        temp = self._data['y0'] + self._data[('eff', str(w))]
        temp.columns = self.targets.columns
        return temp

    @property
    def features(self):
        return self._data['features']

    @staticmethod
    def new_instance(conf: ConfigTree):
        data_conf = conf.get('dataset', conf)
        data = Sim2Dataset(conf=conf)
        if 'pickle' == data_conf.get('data.type'):
            data = pickle.load(open(data_conf.get('data.path'), 'rb'))
            data.conf = conf
        elif 'csv' == data_conf.get('data.type'):
            data.read(data_conf.get('data.path'))
        else:
            raise ValueError(f'unknown data type {conf.get("data.type")}')
        data.description()
        return data

    def dummy_variabe(self):
        I_w = np.eye(self.n_treatment)
        I_t = np.eye(self.treat_dt)
        dummy_w = I_w[self.treatment, :]
        dummy_t = I_t
        return dummy_w, dummy_t

    def dummy_zmulz(self, index=None, leaves_range=None):
        """calculate `\\sum_{i} z_i z_i^T` for each leaves.

        Args:
            index (_type_, optional): The index of each instance. Defaults to None.
            leaves_range (_type_, optional): Indicating mapping of instances and leaves. Each term of `leaves_range` is 
                tuple of range like, [st_pos, end_pos), which means from st_pos to end_pos in array `index` are in the same
                leaf. Defaults to None.
        Returns:
            _type_: _description_
        """
        n_w = self.n_treatment
        t0 = self.treat_dt
        if index is None:
            zz = np.eye(n_w + t0)
            cnt_w = np.zeros([n_w], np.int32)
            countby(self.treatment, [self.treatment], cnt_w)
            zz[np.diag_indices(n_w)] = cnt_w * t0
            zz[(range(n_w, n_w + t0), range(n_w, n_w + t0))] = cnt_w.sum()
            zz[:n_w, n_w:n_w + t0] = np.repeat(cnt_w[:, np.newaxis], t0, axis=1)
            zz[n_w:n_w + t0, 0:n_w] = zz[:n_w, n_w:n_w + t0].T
        else:
            n, m = self.features.shape
            l = len(leaves_range)
            n_bins = 64
            out = np.zeros([l, m, n_w, n_bins], dtype=np.float)
            target = np.ones([n, 1], dtype=np.float)
            update_histogram(target, self.bin_features.to_numpy(np.int32), index, leaves_range,
                             self.treatment.to_numpy(np.int32), out)
            # for [l, m, bin] zz
            ZZ = np.tile(np.eye(n_w + t0), [l, m, n_bins, 1, 1])
            for i in range(l):
                for j in range(m):
                    for k in range(n_bins):
                        cnt_w = out[i, j, :, k]
                        ZZ[i, j, k][np.diag_indices(n_w)] = cnt_w * t0
                        ZZ[i, j, k, (range(n_w, n_w + t0), range(n_w, n_w + t0))] = cnt_w.sum()
                        ZZ[i, j, k, :n_w, n_w:n_w + t0] = np.repeat(cnt_w[:, np.newaxis], t0, axis=1)
                        ZZ[i, j, k, n_w:n_w + t0, 0:n_w] = ZZ[i, j, k, :n_w, n_w:n_w + t0].T
            zz = ZZ
        return zz

    def dummy_ymulz(self, index=None, leaves_range=None):
        n_w = self.n_treatment
        t0 = self.treat_dt
        T = self.n_period
        sumy = np.zeros([2, self.n_period], np.float64)
        sumby(self.targets.to_numpy(), [self.treatment.astype(np.int32).to_numpy()], sumy)
        return np.concatenate([np.sum(sumy, 1), np.sum(sumy, 0)])


class Sim3Dataset(Sim2Dataset):
    def read(self, filenames):
        if len(filenames) == 1:
            unit = pd.read_csv(filenames[0], header=0, index_col=0)
        else:
            raise ValueError(f'')
        # get y columns
        y_cols = [c for c in unit.columns if c.startswith('Y_')]
        x_cols = [c for c in unit.columns if c.startswith('x')]
        w_col = self.info.treatment_column
        unit = unit[x_cols + [w_col] + y_cols]
        [('y', i) for i, _ in enumerate(y_cols)] + [(i, c) for i, c in enumerate(y_cols)]
        new_cols = []
        for c in unit.columns:
            if c in y_cols:
                new_cols.append(('y', y_cols.index(c)))
            elif c in x_cols:
                new_cols.append(('features', c))
            elif c in [w_col]:
                new_cols.append(('treatment', c))
            else:
                new_cols.append(('other', c))
        unit.columns = pd.MultiIndex.from_tuples(new_cols)
        unit[('eff', 1)] = 3
        self._data = unit
        self.treatment = self._data[('treatment', w_col)].astype(np.int32)
        self.treatment.name = 'treatment'
        self.n_period = self.targets.shape[1]
        self.n_inst = self.targets.shape[0]
        self.n_feat = self.features.shape[1]

    @staticmethod
    def new_instance(conf: ConfigTree):
        data_conf = conf.get('dataset', conf)
        data = Sim3Dataset(conf=conf)
        if 'csv' == data_conf.get('data.type'):
            data.read(data_conf.get('data.path'))
        else:
            raise ValueError(f'unknown data type {conf.get("data.type")}')
        data.description()
        return data


if __name__ == '__main__':
    from histogram import Histogram

    conf = ConfigFactory.parse_file('config/didforest.conf')
    d = Sim3Dataset.new_instance(conf)
    name = conf.get('dataset.name')
    # export encoder
    path = f'data/{name}'
    if os.path.exists(path) is False:
        os.mkdir(path)
    # calculate histogram
    h = Histogram(conf)
    h.binning(d)
    n = d.features.shape[0]
    index = np.arange(n, dtype=np.int32)
    leaves_range = np.array([[0, 5000], [5000, n]], np.int32)
    print(d.dummy_ymulz().shape, d.dummy_zmulz(index, leaves_range))

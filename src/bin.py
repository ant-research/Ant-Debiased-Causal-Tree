# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

from pyhocon import ConfigTree
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from aggregate import Value2BinParallel, FindBinParallel
from information import GBCTDataInfo


class BinMapper(KBinsDiscretizer):

    def __init__(self, conf: ConfigTree):
        self.info = GBCTDataInfo(conf)
        self._binmaper_cpp: list = None

    def transform(self, X):
        return Value2BinParallel(X, self._binmaper_cpp)

    def fit(self, X, y=None):
        xshape = X.shape
        assert len(xshape) == 2, f'`X` must be 2-dimension!'
        self._binmaper_cpp = FindBinParallel(X, self.info.n_bins, self.info.min_point_per_bin,
                                             self.info.min_point_per_bin, True)
        return self

    def inverse_transform(self, Xt, index: int = None):
        if index is not None:
            assert len(self._binmaper_cpp) > index and index >= 0, f'index must between [0, {len(self._binmaper_cpp)})!'
            return self._binmaper_cpp[index].BinToValue(Xt)
        raise NotImplementedError

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)
    
    def fit_dataset(self, data):
        x = np.ascontiguousarray(data.features.to_numpy())
        if self.is_fit is False:
            self.fit(x)
        bin_features = self.transform(x)
        bin_features = pd.DataFrame(bin_features, columns=data.features.columns)
        data.bin_features = bin_features

    @property
    def is_fit(self):
        return self._binmaper_cpp is not None

    @property
    def upper_bounds(self):
        return np.asfarray([m.GetUpperBoundValue() for m in self._binmaper_cpp])

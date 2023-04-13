# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

import os
import pickle
from typing import NoReturn
import logging

import numpy as np
import pandas as pd
from pyhocon import ConfigTree

from aggregate import update_histogram
from dataset import Dataset
from information import GBCTDataInfo


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Histogram(object):

    def __init__(self, conf: ConfigTree):
        hist_conf = conf.get('histogram', conf)
        self.conf = conf
        self.info = GBCTDataInfo(conf)
        self.tr_dts = []
        self.max_bin_num = hist_conf.max_bin_num  # Maximum number of bins
        self.min_point_per_bin = hist_conf.min_point_per_bin  # Minimum number of points for binning
        # for accelerate find optimal split point
        self.bin_sum = {}
        # [leaf, feature, treatment, bin, target]
        self.bin_counts = None
        self.bin_y_hist = None
        self.bin_ysq_hist = None
        self.bin_grad_hist = None
        self.bin_hess_hist = None
        self.bin_cgrad_hist = None
        self.bin_chess_hist = None
        #
        self._data = None  # pd.concat({'treatment': treatment, 'target': target}, axis=1)

    @staticmethod
    def binning_feature(features, upper_bound, treatment, col):
        feature = features[col]
        bin_features = pd.cut(feature, [-np.Inf] + list(upper_bound), labels=False)
        return bin_features

    def update_hists(self, target, index, leaves_range, treatment, bin_features, is_gradient=False):
        """update histograms for all nodes in the same level of a tree

        Args:
            data (Dataset):
            index (Matrix): [n, ]. 
            gradients (Dict, optional): _description_. Defaults to None.
            cf_gradients (Dict, optional): _description_. Defaults to None.
        """
        n, m = bin_features.shape
        n_w = 2
        l = len(leaves_range)

        n_bins = self.max_bin_num
        if is_gradient:
            if len(target) == 2:
                (g, h), (cg, ch) = target
            else:
                raise ValueError(f'parameter target must be a two elements tuple')
            n_y = g.shape[1]
            # update histogram of g,h
            out = np.zeros([l, m, n_bins, n_w, n_y], g.dtype)
            update_histogram(g, bin_features, index, leaves_range, treatment, out, n_w, n_bins)
            self.bin_grad_hist = out
            out = np.zeros([l, m, n_bins, n_w, n_y], g.dtype)
            update_histogram(h, bin_features, index, leaves_range, treatment, out, n_w, n_bins)
            self.bin_hess_hist = out
            if cg is not None:
                # update histogram of cg, ch
                out = np.zeros([l, m, n_bins, n_w, n_y], g.dtype)
                update_histogram(cg, bin_features, index, leaves_range, treatment, out, n_w, n_bins)
                self.bin_cgrad_hist = out
                out = np.zeros([l, m, n_bins, n_w, n_y], g.dtype)
                update_histogram(ch, bin_features, index, leaves_range, treatment, out, n_w, n_bins)
                self.bin_chess_hist = out
        else:
            n_y = target.shape[1]
            # update histogram of target
            out = np.zeros([l, m, n_bins, n_w, n_y], target.dtype)
            update_histogram(target, bin_features, index, leaves_range, treatment, out, n_w, n_bins)
            self.bin_y_hist = out
            # update histogram of square target
            out = np.zeros([l, m, n_bins, n_w, n_y], target.dtype)
            update_histogram(target**2, bin_features, index, leaves_range, treatment, out, n_w, n_bins)
            self.bin_ysq_hist = out
        # update counts
        out = np.zeros([l, m, n_bins, n_w, 1], np.int32)
        update_histogram(np.ones([n, 1], np.int32), bin_features, index, leaves_range, treatment, out, n_w, n_bins)
        self.bin_counts = out[:, :, :, :, 0]
        return self

    def dump(self, path: str) -> NoReturn:
        d = {}
        d['bin_mapper'] = self.bin_mapper
        d['min_point_per_bin'] = self.min_point_per_bin
        d['max_bin_num'] = self.max_bin_num
        dir = os.path.dirname(path)
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        file = open(path, 'wb')
        pickle.dump(d, file)

    @classmethod
    def new_instance(cls, dataset: Dataset, conf: ConfigTree = None, **kwargs):
        conf_hist = conf.get('histogram', conf)
        path = conf_hist.get('path', None)
        if conf_hist.load and path and os.path.exists(path):
            bins = pickle.load(open(path, 'rb'))
            hist = cls(conf, dataset.treatment, dataset.targets, bins['bins'])
            logger.debug(f'load histogram from file: {path}!')
        else:  #  calculate histogram
            hist = cls(conf, dataset.treatment, dataset.targets)
        hist.binning(dataset)
        return hist

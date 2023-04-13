# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

from pyhocon import ConfigTree


class DataInfo(object):
    def __init__(self, conf, **kwargs):
        self.names = conf.get('dataset.name')
    
    def description(self,):
        pass


class CausalDataInfo(DataInfo):
    def __init__(self, conf, **kwargs):
        super(CausalDataInfo, self).__init__(conf, **kwargs)
        data_conf= conf.get('dataset', conf)
        self.n_treatment = data_conf.get('n_treatment')
        self.feature_columns = data_conf.get('feature')
        self.treatment_column = data_conf.get('treatment')
        self.feature_ratio = conf.get('feature_ratio')
        self.instance_ratio = conf.get('instance_ratio')

        hist_conf = conf.get('histogram')
        self.n_bins = hist_conf.get('max_bin_num')
        self.min_point_per_bin = hist_conf.get('min_point_per_bin')
        
        tree_conf = conf.get('tree')
        self.min_point_num_node = tree_conf.get('min_point_num_node')
        self.max_depth = tree_conf.get('max_depth')


class GBCTDataInfo(CausalDataInfo):
    def __init__(self, conf: ConfigTree, **kwargs):
        super(GBCTDataInfo, self).__init__(conf, **kwargs)
        data_conf= conf.get('dataset', conf)
        self.n_period = data_conf.get('n_period')
        self.treat_dt = data_conf.get('treat_dt')
        
        tree_conf = conf.get('tree')
        self.lambd = tree_conf.get('lambd')
        self.gamma = tree_conf.get('gamma')
        self.coef = tree_conf.get('coefficient')

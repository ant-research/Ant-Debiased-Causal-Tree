# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

import pickle
from typing import NoReturn
import logging

import numpy as np
import pandas as pd

from reflect_utils import get_class

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self):
        pass

    def __len__(self):
        return self.features.shape[0]

    @staticmethod
    def new_instance(conf):
        data_conf = conf.get('dataset', conf)
        cls_name = data_conf.get('type', 'dataset.CSVDataset')
        return get_class(cls_name).new_instance(conf)

    def read(self, filename):
        pass

    def sub_dataset(self, index = None):
        raise NotImplementedError

    def head(self, n):
        return self.sub_dataset(pd.Series(np.arange(n)))

    def get_pretreated_targets(self):
        pass

    def description(self, detail: bool = False) -> None:
        """description the dataset

        Args:
            detail (bool, optional): [description]. Defaults to False.
        """
        n_ins, n_feat = self.features.shape
        n_y_len = self.targets.shape[1]
        # calculate treatment distinct count
        treats = np.unique(self.treatment)
        logger.info(f'#inst: {n_ins}')
        logger.info(f'#feat: {n_feat}')
        logger.info(f'#time serise length: {n_y_len}')
        logger.info(f'#treatments : {len(treats)}')
        if detail:
            pass

    def split(self, col, bind):
        raise NotImplementedError

    def save(self, path: str = None) -> NoReturn:
        if path is None:
            path = self.conf.path
        pickle.dump({
            'data': self._data,
            'targets': self.targets,
            'treatment': self.treatment
        },
                    open(path, 'wb'),
                    protocol=4)

    def read_pickle(self, path: str = None) -> NoReturn:
        tmp = pickle.load(open(path, 'rb'))
        assert 'data' in tmp and 'targets' in tmp
        if isinstance(tmp['data'], pd.DataFrame):
            _data = tmp['data']
            targets = tmp['targets']
        else:
            _data = pd.DataFrame(tmp['data'])
            targets = tmp['targets']
        self.targets = targets.astype(np.float32)
        self.features = _data[self.conf.feature]
        self.treatment = _data[self.conf.treatment].apply(lambda x: max(x, 0))
        self.n_inst = self.features.shape[0]
        self.n_feat = self.features.shape[1]

    def parse(self):
        raise NotImplementedError

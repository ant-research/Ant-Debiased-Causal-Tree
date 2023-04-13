# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
import logging
import os
import pickle
from time import time

import numpy as np
from pyhocon import ConfigTree, ConfigFactory
from bin import BinMapper
from gradient_did_tree import GradientDebiasedCausalTree
from dataset import Dataset
from information import GBCTDataInfo
from cppnode import predict


class Boosting(object):
    def __init__(self, tree_cls, conf: ConfigTree, bin_mapper: BinMapper = None):
        """_summary_

        Args:
            tree_cls (_type_): _description_
            conf (ConfigTree): parameter's configure. For details of the parameters, refer to the the files in config.
            bin_mapper (BinMapper, optional): _description_. Defaults to None.
        """
        self.info = GBCTDataInfo(conf)
        self.conf = conf
        self.n_estimators = conf.iterators
        self.trees = []
        self.subfeature = conf.feature_ratio
        self.subsample = conf.instance_ratio
        self.learning_rate = conf.shrinkage
        self.valid_losses = []
        self.bin_mapper = bin_mapper if bin_mapper is not None else BinMapper(conf)
        self.verbose = False
        self.tree_cls = tree_cls
        self.n_iter_no_change = 10
        self.tol = 1e-3
        self.opt_step = 0

    def fit(self, data: Dataset, test_data: Dataset = None):
        """fitting

        Args:
            data (Dataset): the training dataset
            test_data (Dataset, optional): the test dataset. Defaults to None.
        """
        st = time()
        self.preprocess(data)
        pred = np.zeros_like(data.targets)
        cpred = np.zeros_like(data.targets)
        target = data.targets.to_numpy()
        features = np.ascontiguousarray(data.features.to_numpy())
        w = np.ascontiguousarray(data.treatment.to_numpy(np.int32))
        cur_pred = np.zeros_like(data.targets, dtype=target.dtype)
        cur_cpred = np.zeros_like(data.targets, dtype=target.dtype)
        print_intervels = 1
        for i in range(self.n_estimators):
            tr_data, _, idx, val_idx = self.tr_val(data)
            if i % print_intervels == 0:
                logging.info(f'{"="*20}{i}-th tree{"="*20}')
            tree = self.tree_cls(self.conf, self.bin_mapper, verbose=self.verbose)
            g, h = tree.gradients(target, pred)
            cg, ch = tree.gradients(target, cpred)
            tree.fit((g[idx], h[idx]), (cg[idx], ch[idx]), tr_data)
            cur_pred, cur_cpred, _ = tree.predict(features, w, key='cf_outcomes', out=(cur_pred, cur_cpred))
            pred += self.learning_rate * cur_pred
            cpred += self.learning_rate * cur_cpred
            # 按照w来平衡
            preloss = np.square(target[:, :-1] - cpred[:, :-1])[val_idx]
            postloss =np.square(target[:, -1:] - pred[:, -1:])[val_idx]

            preloss = np.sqrt((preloss[w[val_idx]==1].mean() + preloss[w[val_idx]==0].mean())/2)
            postloss = np.sqrt((postloss[w[val_idx]==1].mean() + postloss[w[val_idx]==0].mean())/2)

            self.valid_losses.append(preloss + self.conf.tree.coefficient * postloss)
            if i % print_intervels == 0:
                logging.info(f'Debias loss:{preloss:.3f}\tpostloss:{postloss:.3f}\ttotal:{self.valid_losses[-1]:.3f}')
            self.trees.append(tree)
            if i % print_intervels == 0 and test_data is not None:
                self.predict(test_data)
            if self.early_stopping():
                logging.info(f'early stop')
                break
        logging.info(f'time exhausted: {time() - st:.2f}s')

    def preprocess(self, data: Dataset):
        self.bin_mapper.fit_dataset(data)

    def tr_val(self, data: Dataset):
        # split
        n = len(data)
        tr_n = int(n * self.info.instance_ratio)
        idx = np.random.permutation(n).astype(np.int32)
        if tr_n < n:
            idx = np.random.permutation(n).astype(np.int32)
            return data.sub_dataset(idx[:tr_n]), data.sub_dataset(idx[tr_n:]), idx[:tr_n], idx[tr_n:]
        else:
            return data, data, slice(None), slice(None)

    def postprocess(self, ):
        opt_n = np.argmin(self.valid_losses) + 1
        self.trees = self.trees[:opt_n]

    def early_stopping(self) -> bool:
        # use both pre_loss and post_loss
        no_change_steps = self.n_iter_no_change if np.isscalar(self.n_iter_no_change) else np.inf
        n = len(self.valid_losses)
        min_loss = self.valid_losses[self.opt_step]
        cur_loss = self.valid_losses[-1]
        if (min_loss - cur_loss)/min_loss >= self.tol: 
            self.opt_step = np.argmin(self.valid_losses)
        if no_change_steps <= n - self.opt_step - 1:
            return True
        if len(self.trees) > 0 and self.trees[-1].root.is_leaf:
            logging.info(f'The last tree is no more splitting!')
            return True
        if n - self.opt_step > 1:
            logging.info(f'{self.opt_step}-th has min loss ({min_loss:.3}), no change steps: {n - self.opt_step - 1}')
        return False

    def predict_leaves(self, data: Dataset):
        x = data.features.to_numpy()
        if x.flags.c_contiguous is False:
            x = np.ascontiguousarray(x)
        leaf_ids = np.zeros([x.shape[0], len(self.trees)], dtype=np.float)
        predict([tree.export() for tree in self.trees], x, leaf_ids, "leaf_id")
        return leaf_ids

    def predict(self, data: Dataset):
        x = data.features.to_numpy()
        w = data.treatment.to_numpy()
        if x.flags.c_contiguous is False:
            x = np.ascontiguousarray(x)
            w = np.ascontiguousarray(w)
        results = np.zeros([x.shape[0], len(self.trees), 1, 8], dtype=np.float64)
        predict([tree.export() for tree in self.trees], x, results, "debiased_effect")
        tau = data._data['eff', 1].to_numpy()
        tau_hat = self.learning_rate * results.sum(axis=1)
        tau_hat = tau_hat[:, 0, self.info.treat_dt] - tau_hat[:, 0, :self.info.treat_dt].mean(axis=1)
        logging.info(f'PEHE: {np.sqrt(np.square(tau_hat-tau).mean()):.3f}, {np.sqrt(np.square(tau-tau.mean()).mean()):.3f}')
        logging.info(f'MAE_HTE: {(abs(tau_hat-tau).mean()):.3f}, {(abs(tau-tau.mean()).mean()):.3f}')
        return tau_hat


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', action='store', help='configure file', default='config/gbct.conf')
    parser.add_argument('-o', '--out', action='store', help='model name prefix')
    parser.add_argument('-e', '--coefficient', action='store', type=float, help='post-treat loss coefficient')
    parser.add_argument('-d', '--data_path', action='store', help='data path', default=None)
    args = parser.parse_args()
    conf_path = args.conf
    model_name = args.out
    np.random.seed(0)
    conf = ConfigFactory.parse_file(conf_path)
    if args.data_path is not None:
        data_conf = ConfigFactory.parse_file(args.data_path)
        p = os.path.dirname(data_conf.data.path[0])
        data_conf.put('$.data.path', [f'{p}/X.csv', f'{p}/Y.csv'])
        conf.put('dataset', data_conf)
    if args.coefficient is not None:
        conf.put('tree.coefficient', args.coefficient)
    data = Dataset.new_instance(conf)
    bin_mapper = BinMapper(conf)
    idx = np.load(os.path.dirname(conf.dataset.data.path[0])+'/tr_te.npy', allow_pickle=True)
    for i in range(idx.shape[0]):
        cf = Boosting(GradientDebiasedCausalTree, conf, bin_mapper)  # type: ignore
        logging.info(f'{"="*20}fold-{i}{"="*20}')
        tr_data = data.sub_dataset(idx[i, 0])
        valid_data = data.sub_dataset(idx[i, 1])
        cf.fit(data, valid_data)
        print(cf.predict(valid_data).mean())
        dir = f'{os.path.dirname(conf.dataset.data.path[0])}/fold-{i}/'
        if os.path.exists(dir) is False:
            os.mkdir(dir)
        with open(f'{dir}/{model_name}.pkl', 'wb') as f:
            pickle.dump(cf, f)

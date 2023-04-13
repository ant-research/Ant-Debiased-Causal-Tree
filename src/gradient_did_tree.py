# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
from functools import partial
from typing import Dict, List
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

from pyhocon import ConfigTree
import numpy as np

from did_node import GradientCausalTreeNode
from dataset import Dataset
from histogram import Histogram
from information import GBCTDataInfo
from aggregate import update_x_map
from cppnode import create_didnode_from_dict, predict, indexbyarray
from losses import Loss


gamma = 0

def _filter(leaves, leaves_range):
    # 过滤掉停止分裂的节点
    inner_leaves_range = []
    inner_leaves = []
    for i, leaf in enumerate(leaves):
        if leaf.is_leaf is False:
            inner_leaves_range.append(leaves_range[i])
            inner_leaves.append(leaf)
    if len(inner_leaves_range) > 0:
        inner_leaves_range = np.stack(inner_leaves_range, axis=0)
    else:
        return [], None
    return inner_leaves, inner_leaves_range


def mix_losses(sum_loss, cnt, coeff=1, use_ps=True, treated_ts=-1, **kwargs):
    loss = 0
    eta = kwargs.get('eta', 0)
    if use_ps:
        loss = np.mean(sum_loss / cnt, axis=0) * cnt.sum()
    else:
        loss = np.sum(sum_loss, axis=0)
    return loss[:treated_ts].mean() + coeff * loss[treated_ts:].sum() + .5*gamma * eta[0]**2 *cnt.sum()


def _estimate(configs, hist: Histogram, obj):
    # 估计所有的情况的theta
    losses = {}
    lambd = obj.info.lambd
    coeff = obj.info.coef
    t0 = obj.info.treat_dt
    est_fn = obj.root.estimate
    loss_fn = obj.root.fast_loss
    n_leaf, n_feat, n_bins, n_w, n_y = hist.bin_grad_hist.shape
    # [n_leaf, n_features, n_bins, n_treatment, n_outcome]
    Gs = hist.bin_grad_hist.sum(axis=2)
    Hs = hist.bin_hess_hist.sum(axis=2)
    CGs = hist.bin_cgrad_hist.sum(axis=2)
    CHs = hist.bin_chess_hist.sum(axis=2)
    CNT = np.expand_dims(hist.bin_counts[:, 0].sum(axis=1), axis=-1)
    for level_id, leaf_info in configs.items():
        # [n_leaf, n_feature, n_bin, n_w + t0, n_w + t0]
        candi_feats = {}
        for fid, feature_info in leaf_info.items():
            candi_bins = []
            l_grad = np.zeros([n_w, n_y])
            l_hess = np.zeros([n_w, n_y])
            l_cgrad = np.zeros([n_w, n_y])
            l_chess = np.zeros([n_w, n_y])
            l_cnt = np.zeros([n_w, 1], np.int32)
            cur_bin_idx = 0
            for b in range(n_bins):
                l_cnt += np.expand_dims(hist.bin_counts[level_id, fid, b], axis=-1)
                l_grad += hist.bin_grad_hist[level_id, fid, b]
                l_hess += hist.bin_hess_hist[level_id, fid, b]
                l_cgrad += hist.bin_cgrad_hist[level_id, fid, b]
                l_chess += hist.bin_chess_hist[level_id, fid, b]
                if cur_bin_idx < len(feature_info) and feature_info[cur_bin_idx] == b:
                    cur_bin_idx += 1
                    # left node parameter estimation
                    l_f_yhat = est_fn(l_grad, l_hess, lambd=lambd)
                    l_c_yhat = est_fn(l_grad.sum(0, keepdims=True) - l_grad,
                                      l_hess.sum(0, keepdims=True) - l_hess,
                                      lambd=lambd)
                    l_eta, l_yhat, (l_grad_hat, l_hess_hat) = obj.cross((l_grad, l_hess), (l_cgrad, l_chess), l_c_yhat,
                                                                        l_f_yhat, l_cnt, t0, lambd)
                    lloss = loss_fn(y_hat=l_yhat, grad=l_grad_hat, hess=l_hess_hat, lambd=lambd)
                    # right node parameter estimation
                    r_grad, r_hess = Gs[level_id, fid] - l_grad, Hs[level_id, fid] - l_hess
                    r_cgrad, r_chess = CGs[level_id, fid] - l_cgrad, CHs[level_id, fid] - l_chess
                    r_f_yhat = est_fn(r_grad, r_hess, lambd=lambd)
                    r_c_yhat = est_fn(r_grad.sum(0, keepdims=True) - r_grad,
                                      r_hess.sum(0, keepdims=True) - r_hess,
                                      lambd=lambd)
                    r_cnt = CNT[level_id] - l_cnt
                    r_eta, r_yhat, (r_grad_hat, r_hess_hat) = obj.cross((r_grad, r_hess), (r_cgrad, r_chess), r_c_yhat,
                                                                        r_f_yhat, r_cnt, t0, lambd)
                    rloss = loss_fn(y_hat=r_yhat, grad=r_grad_hat, hess=r_hess_hat, lambd=lambd)
                    _loss = mix_losses(lloss, l_cnt, coeff=coeff, eta=l_eta) + mix_losses(
                        rloss, r_cnt, coeff=coeff, eta=r_eta)
                    candi_bins.append((b, _loss, l_f_yhat, r_f_yhat, l_eta, r_eta))
            if len(candi_bins) > 0:
                candi_feats[fid] = candi_bins
        if len(candi_feats) > 0:
            losses[level_id] = candi_feats
        else:
            if obj.verbose:
                print(f'the {level_id}-th node in this level stop splitting!')
    return losses


class GradientDebiasedCausalTree:
    def __init__(self, conf: ConfigTree = None, bin_mapper=None, **kwargs):
        self.conf = conf
        self.info = GBCTDataInfo(conf)
        self.verbose = kwargs.get('verbose', False)
        self.feature_used = []
        self.feature_used_map = {}  # key: sub-feature index, value: original feature index
        self.bin_mapper = bin_mapper
        conf_tree = conf.get('tree', conf)
        self.op_loss = Loss.new_instance(conf_tree)
        self.did = False

    def fit(self, gradients, cgradients, data: Dataset):
        hist, idx_map = self.preprocess(gradients, cgradients, data, self.info.instance_ratio, self.info.feature_ratio)
        root = GradientCausalTreeNode(self.conf, leaf_id=0, level_id=0)
        self.root = root
        leaves = [root]
        leaves_range = np.array([[0, self.inst_used]], np.int32)
        # calculate loss
        leaf_id = root.leaf_id
        gsum = hist.bin_grad_hist[leaf_id, 0].sum(0)
        hsum = hist.bin_hess_hist[leaf_id, 0].sum(0)
        root.theta = root.estimate(gsum, hsum, lambd=self.info.lambd)
        eta = np.zeros([self.info.n_treatment,1])
        self.root.eta =eta
        for i in range(self.info.max_depth):
            if self.verbose:
                print(f'{"--"*10} the {i}-th iterations {"--"*10}')
            split_conds = self.split(leaves, hist)
            leaves, leaves_range = self.updater(split_conds, gradients, cgradients, data, hist, idx_map, leaves,
                                                leaves_range)
            if len(leaves) == 0:
                print(f'early stop!')
                break
        for leaf in leaves:
            leaf.is_leaf = True
        self.postprocess()

    def updater(self, split_conds: Dict, gradients, cgradients, tr_data, hist: Histogram, idx_map,
                leaves: List[GradientCausalTreeNode], leaves_range):
        leaves, leaves_range = _filter(leaves, leaves_range)
        if len(leaves) == 0:
            return leaves, leaves_range
        n_leaf = len(split_conds)
        x_binned = tr_data.bin_features[self.feature_used].to_numpy(np.int32)
        treatment = tr_data.treatment.to_numpy(np.int32)
        sorted_split = OrderedDict(sorted(split_conds.items()))
        split_info = np.asfarray([[info['feature'], info['threshold']]
                                  for _, info in sorted_split.items()]).astype(np.int32)
        out = np.zeros([n_leaf * 2, 2], np.int32)
        update_x_map(x_binned, idx_map, split_info, leaves_range, out)
        leaves_range_new = out
        hist.update_hists((gradients, cgradients), idx_map, leaves_range_new, treatment, x_binned, is_gradient=True)
        # create new node
        leaves_new = []
        for i, leaf in enumerate(leaves):
            ltheta, rtheta = split_conds[leaf.level_id]['theta']
            l_eta, r_eta = split_conds[leaf.level_id]['eta']
            leaf._children = [
                GradientCausalTreeNode(self.conf,
                                          leaf_id=leaf.leaf_id * 2 + 1,
                                          level_id=i * 2,
                                          theta=ltheta,
                                          eta=l_eta),
                GradientCausalTreeNode(self.conf,
                                          leaf_id=leaf.leaf_id * 2 + 2,
                                          level_id=i * 2 + 1,
                                          theta=rtheta,
                                          eta=r_eta)
            ]
            leaves_new.extend(leaf._children)
            fid, bin_id = split_info[i]
            leaf.split_feature = self.feature_used_map[fid]
            leaf.split_thresh = bin_id
            leaf.split_rawthresh = self.bin_mapper.inverse_transform(bin_id, self.feature_used_map[fid])
        return leaves_new, leaves_range_new

    def cross(self, grads, cgrads, c_yhat, f_yhat, cnt, t0, lambd):
        grad, hess = grads
        cgrad, chess = cgrads
        eta = np.zeros([self.info.n_treatment,1])
        grad_hat = np.concatenate([cgrad[:, :t0], grad[:, t0:]], axis=-1)
        hess_hat = np.concatenate([chess[:, :t0], hess[:, t0:]], axis=-1)
        yhat = np.concatenate([c_yhat[:, :t0], f_yhat[:, t0:]], axis=-1)
        return eta, yhat, (grad_hat, hess_hat)

    def split(self, leaves: List[GradientCausalTreeNode], hist: Histogram):
        # 分裂本层的叶子节点
        # step 1, 收集所有需要计算损失的分裂点
        # step 2,
        info = self.info
        n_leaves, m, n_bins, n_w, n_y = hist.bin_grad_hist.shape
        t0, T, n_w = info.treat_dt, info.n_period, info.n_treatment
        min_num = self.info.min_point_num_node
        configs = {}  # {leaf:{feature:[bin]}}
        parallel_configs = []
        # 生成所以可能的分裂组合
        for leaf in leaves:
            level_id = leaf.level_id
            candi_feats = {}
            cnt_sum = np.sum(hist.bin_counts[level_id, 0], axis=0)
            for fid in range(m):
                candi_bins = []
                cnt = np.zeros([n_w], np.int32)
                for b in range(n_bins):
                    cnt += hist.bin_counts[level_id, fid, b]
                    if all(cnt >= min_num) and all((cnt_sum - cnt) >= min_num):
                        candi_bins.append(b)
                candi_feats[fid] = candi_bins
                parallel_configs.append({level_id: {fid: candi_bins}})
            configs[level_id] = candi_feats

        with ProcessPoolExecutor() as pool:
            losses = {}
            for r in pool.map(partial(_estimate, hist=hist, obj=self), parallel_configs):
                for level_id in r:
                    if level_id not in losses:
                        losses[level_id] = {}
                    for fid in r[level_id]:
                        if fid not in losses[level_id]:
                            losses[level_id][fid] = []
                        losses[level_id][fid].extend(r[level_id][fid])
        # losses = _estimate(configs, hist, self)
        split_conds = {}
        for leaf in leaves:
            level_id = leaf.level_id
            if level_id not in losses:
                leaf.is_leaf = True
                continue
            leaf_losses = losses[level_id]
            _loss, _fid, _thresh = np.inf, None, None
            for fid, feature_info in leaf_losses.items():
                for b, loss, l_theta, r_theta, l_eta, r_eta in feature_info:
                    if loss < _loss:
                        _loss, _fid, _thresh, _theta, _eta = loss, fid, b, (l_theta, r_theta), (l_eta, r_eta)
            if self.verbose:
                fid_ = self.feature_used_map[_fid]
                print(f'leaf {leaf.leaf_id}, best split feature {fid_} bin({_thresh}) \tand loss:\t {_loss:.3f}')
            leaf._loss = _loss
            assert level_id not in split_conds
            split_conds[level_id] = {'feature': _fid, 'threshold': _thresh, 'loss': _loss, 'theta': _theta, 'eta': _eta}
        return split_conds

    def preprocess(self, gradients, cgradients, tr_data: Dataset, subsample=1, subfeature=1):
        n, m = tr_data.features.shape
        index = np.random.permutation(n).astype(np.int32)
        n_used, m_used = np.math.ceil(n * subsample), np.math.ceil(m * subfeature)

        features = self.info.feature_columns
        # subsampling
        if m_used < m:
            tmp_feat = np.random.choice(features, m_used, replace=True)
            features = [f for f in features if f in tmp_feat]
        else:
            features = self.info.feature_columns
        hist = Histogram(self.conf)
        if tr_data.bin_features is None:
            self.bin_mapper.fit_dataset(tr_data)
        else:
            hist.columns = list(tr_data.features.columns)
        x_binned = np.ascontiguousarray(tr_data.bin_features[features].to_numpy(np.int32))
        self.feature_used = features
        self.inst_used = n_used
        orig_features = list(tr_data.features.columns)
        self.feature_used_map = {i: orig_features.index(f) for i, f in enumerate(features)}
        # calculate histogram for outcome
        w = np.ascontiguousarray(tr_data.treatment.to_numpy(np.int32))
        hist.update_hists((gradients, cgradients), index, np.array([[0, n_used]], np.int32), w, x_binned, True)
        return hist, index

    def export(self):
        nodes, queue = [], [self.root]
        while len(queue) > 0:
            nodes.append(queue.pop(0))
            for child in nodes[-1].children:
                queue.append(child)
        # encode for each node
        slim_nodes = []
        t_0 = self.info.treat_dt
        for child in nodes:
            bias, effect, debiased_effect = [], [], []
            for _w in range(1, self.info.n_treatment):
                bias.append(np.mean(child.theta[_w, :t_0] - child.theta[0, :t_0]))
                effect.append(child.theta[_w] - child.theta[0])
                debiased_effect.append(effect[-1] + child.eta[_w])
            info = {
                'leaf_id': child.leaf_id,
                'level_id': child.level_id,
                'outcomes': child.theta,
                'bias': np.array(bias),
                'eta': child.eta,
                'effect': np.array(effect),
                'debiased_effect': np.array(debiased_effect),
                'is_leaf': child.is_leaf,
                'children': [-1, -1],
                'split_feature': -1,
                'split_thresh': -1
            }
            if child.is_leaf is False:
                info['children'] = [nodes.index(child.children[0]), nodes.index(child.children[1])]
                info['split_feature'] = child.split_feature
                info['split_thresh'] = child.split_rawthresh
            slim_nodes.append(create_didnode_from_dict(info))
        return slim_nodes

    def postprocess(self):
        return self.export()

    def _predict(self, nodes, x, key='effect', out=None):
        assert isinstance(nodes, list) and len(nodes) > 0, f'nodes must be list and at least one element!'
        if key == 'effect':
            shape = (x.shape[0], ) + nodes[0].outcomes.shape
        elif key == 'leaf_id':
            shape = (x.shape[0], 1, 1)
        elif key == 'outcomes':
            shape = (x.shape[0], ) + nodes[0].outcomes.shape
        elif key == 'eta':
            shape = (x.shape[0], 2, 1)
        else:
            raise NotImplementedError
        if x.flags.c_contiguous is False:
            x = np.ascontiguousarray(x)
        if out is None:
            out = np.zeros(shape, np.float64)
        predict(nodes, x, out, key)
        return out

    def predict(self, x, w=None, key='effect', out=None):
        if key == 'cf_outcomes':
            outcome = self.predict(x, key='outcomes')
            cur_pred, cur_cpred = out
            indexbyarray(outcome, w, cur_pred, cur_cpred)
            eta = np.zeros([x.shape[0], self.info.n_treatment, 1])
            return cur_pred, cur_cpred, eta
        return self._predict(self.export(), x, key, out)

    def gradients(self, target, prediction):
        return self.op_loss.gradients(target, prediction)

# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

import numpy as np
from pyhocon import ConfigTree
from losses import Loss


class CausalTreeNode(object):

    def __init__(self, conf: ConfigTree = None, **kwargs):
        conf_tree = conf.get('tree', conf)
        self.leaf_id: int = kwargs.get('leaf_id')
        self.level_id: int = kwargs.get('level_id')
        self.n_period = conf.get('dataset.n_period')
        self.treat_dt = conf.get('dataset.treat_dt')
        self.n_treatment = conf.get('dataset.n_treatment')

        self.is_leaf = False
        self.depth = kwargs.get('depth', 0)
        loss_op = Loss.new_instance(conf_tree)
        self.conf = conf
        self.op_loss: Loss = loss_op
        self.min_point_num_node = conf_tree.min_point_num_node  # Minimum number of points for node splitting
        self.max_depth = conf_tree.max_depth

        self.gamma = kwargs.get('gamma', conf.get('tree.gamma'))
        self.lambd = kwargs.get('lambd', conf.get('tree.lambd'))
        self.coef = kwargs.get('coef', conf.get('tree.coefficient'))
        self.ps_weight = True
        self.gain_thresh = 0
        self._children = []

        self.split_feature = None
        self.split_thresh = None
        self.split_rawthresh = None
        self.theta = kwargs.get('theta', None)

    def predict(self, x):
        # 给定样本做预测
        if self.is_leaf:
            return self.theta

    def estimate(self, zz, yz, yz_cross, cnt):
        # 给定数据，估计当前节点的参数
        t0 = self.treat_dt
        n_w = self.n_treatment
        theta = np.zeros([n_w, self.n_period])
        theta_pre = self._estimate_nuisance(zz, yz[:t0 + n_w], cnt)
        theta_post = self._estimate_interest(yz_cross, cnt)
        for w in range(n_w):
            theta[w, :t0] = theta_pre[n_w:] + theta_pre[w]
            theta[w, t0:] = theta_post[w]
        return theta

    def _estimate_nuisance(self, zz, yz, w_cnt):
        # pre-treatment的参数
        zz_inv = self._inverse_zz(zz, w_cnt)
        return yz @ zz_inv

    def _estimate_interest(self, yz, w_cnt):
        # 估计post-treatment的参数
        treated = self.n_period - self.treat_dt
        return yz / np.repeat(w_cnt, repeats=[treated])

    def _inverse_zz(self, zz, w_cnt):
        lambd = self.lambd
        n_w, t0 = self.n_treatment, self.treat_dt
        a, b, c = 0, n_w, n_w + t0
        A = zz[a:b, a:b] + np.diag(np.full(b - a, lambd))
        A_inv = np.diag(1 / np.diag(A))
        B = zz[a:b, b:c]
        C = zz[b:c, a:b]
        D = zz[b:c, b:c] + np.diag(np.full(c - b, lambd))
        D_inv = np.diag(1 / np.diag(D))
        # print(f'A:{A.shape}, B:{B.shape}, C:{C.shape}, D:{D.shape}')

        alpha = np.sum((w_cnt**2) / (w_cnt * t0 + lambd))
        gamma = D_inv + alpha / ((w_cnt.sum() + lambd) * (w_cnt.sum() + lambd - alpha * t0))
        # assert(np.abs(np.linalg.inv(D-alpha) - gamma).max() < 1e-8)
        CAinv = C @ A_inv
        AinvB = A_inv @ B
        Ainv_B_Gamm = AinvB @ gamma
        Gamm_C_Ainv = gamma @ CAinv
        inv = np.block([[A_inv + Ainv_B_Gamm @ CAinv, -Ainv_B_Gamm], [-Gamm_C_Ainv, gamma]])
        return inv

    def loss(self, y, y_hat, w, weight=None):
        assert weight is None, '`weight` is not supported in current version!'
        assert len(w.shape) == 1 or w.shape[1] == 1, f'multi causes are not supported in current version!'
        total_loss = {}
        loss_fn = self.op_loss.loss
        t_0 = self.treat_dt

        for _w in range(self.n_treatment):
            w_idx = (w == _w)
            lbl = y[w_idx]
            # pre-treatment
            pre_loss = loss_fn(lbl[:, :t_0], y_hat[_w, :t_0])
            # post-treatment
            post_loss = loss_fn(lbl[:, t_0:], y_hat[_w, t_0:])
            total_loss[_w] = sum(np.mean(pre_loss, axis=0)) + self.coef * sum(np.mean(post_loss, axis=0))
        return np.mean(list(total_loss.values())) * y.shape[0]

    @property
    def children(self):
        return self._children


class CausalTreeNode2(CausalTreeNode):

    def __init__(self, conf: ConfigTree = None, **kwargs):
        super().__init__(conf, **kwargs)
        conf_tree = conf.get('tree', conf)
        self.leaf_id: int = kwargs.get('leaf_id')
        self.level_id: int = kwargs.get('level_id')
        self.n_period = conf.get('dataset.n_period')
        self.treat_dt = conf.get('dataset.treat_dt')
        self.n_treatment = conf.get('dataset.n_treatment')

        self.is_leaf = False
        self.depth = kwargs.get('depth', 0)
        loss_op = Loss.new_instance(conf_tree)
        self.conf = conf
        self.op_loss: Loss = loss_op
        self.min_point_num_node = conf_tree.min_point_num_node  # Minimum number of points for node splitting
        self.max_depth = conf_tree.max_depth

        self.gamma = kwargs.get('gamma', conf.get('tree.gamma'))
        self.lambd = kwargs.get('lambd', conf.get('tree.lambd'))
        self.coef = kwargs.get('coef', conf.get('tree.coefficient'))
        self.ps_weight = True
        self.gain_thresh = 0
        self._children = []

        self.split_feature = None
        self.split_thresh = None
        self.split_rawthresh = None
        self.theta = kwargs.get('theta', None)

    def loss(self, y, y_hat, w, weight=None):
        """_summary_

        Args:
            y (_type_): [n_instance, n_outcome]
            y_hat (_type_): [n_treatment, n_outcome]
            w (_type_): [n_instance]
            weight (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        assert weight is None, '`weight` is not supported in current version!'
        assert len(w.shape) == 1 or w.shape[1] == 1, f'multi causes are not supported in current version!'
        total_loss = {}
        loss_fn = self.op_loss.fast_loss
        t_0 = self.treat_dt
        g = self.trends(y[:, :t_0])
        g_hat = self.trends(y_hat[:, :t_0])
        for _w in range(self.n_treatment):
            w_idx = (w == _w)
            # pre-treatment
            pre_loss = loss_fn(g[w_idx, :], g_hat[_w, :])
            # post-treatment
            post_loss = loss_fn(y[w_idx, t_0:], y_hat[_w, t_0:])
            total_loss[_w] = self.coef * np.mean(pre_loss, axis=0) + np.mean(post_loss, axis=0)
        return np.mean(list(total_loss.values()))

    def fast_loss(self, y, y_hat, y_bar=None, y2_mean=None, remove_squared=True):
        return self.op_loss.fast_loss(y, y_hat, y_bar=y_bar, y2_mean=y2_mean, remove_squared=remove_squared)

    def estimate(self, counts, ysum):
        """estimate the theta

        Args:
            counts (_type_): _description_
            ysum (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(counts.shape) == 1:
            counts = np.expand_dims(counts, axis=-1)
        theta = ysum / counts
        return theta
    

class GradientCausalTreeNode(CausalTreeNode2):
    def __init__(self, conf: ConfigTree= None, **kwargs):
        super(GradientCausalTreeNode, self).__init__(conf, **kwargs)
        self.eta = kwargs.get('eta', None)
    
    def approx_loss(self, y_hat, grad, hess, lambd=0, weight=None):
        return self.op_loss.approx_loss((grad, hess), y_hat=y_hat, weight=weight, lambd=lambd)
    
    def fast_loss(self, y_hat, grad, hess, lambd=0, weight=None):
        return self.approx_loss(y_hat, grad, hess, lambd=lambd, weight=weight)
    
    def estimate(self, G, H, **kwargs):
        lambd = kwargs.get('lambd', 0)
        return  -G/(H+lambd)

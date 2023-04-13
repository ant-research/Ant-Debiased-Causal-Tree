# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.

from abc import ABCMeta, abstractmethod
from math import log
from typing import Tuple, Union

import numpy as np
import pandas as pd
from reflect_utils import new_instance

epsilon = np.finfo(np.float32).eps


def sigmoid(x):
    if isinstance(x, (float, int)):
        return 1 / (1 + np.exp(-x))
    elif isinstance(x, (np.ndarray, )):
        return np.apply_along_axis(lambda x: 1 / (1 + np.exp(-np.float32(x))), 0, x)
    else:
        raise ValueError(f'type {type(x)} not supported!')


class Loss(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        self._name = kwargs.get('name', self.__class__.__name__)
        self.classification = True

    @staticmethod
    def new_instance(conf):
        conf = conf.get('tree', conf)
        loss_cls = conf.get('loss_cls', None)
        return new_instance(loss_cls)

    @abstractmethod
    def loss(self, target, prediction, *args):
        """interface of loss

        Args:
            target (Matrix): [description]
            prediction (Matrix): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            Matrix: [description]
        """
        raise NotImplementedError


class GradLoss(Loss):

    @abstractmethod
    def gradients(self, target, prediction) -> Tuple:
        """calculate the gradients and hessians on the `target` and `prediction`

        Args:
            target (DataFrame): [description]
            prediction (DataFrame): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            Union[Tuple, None]: [description]
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, target, prediction):
        """interface of the gradient of loss

        Args:
            target (Matrix): [description]
            prediction (Matrix): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            Matrix: [description]
        """
        raise NotImplementedError

    @abstractmethod
    def hessian(self, target, prediction):
        """interface of the second-step gradient of loss

        Args:
            target (Matrix): [description]
            prediction (Matrix): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            Matrix: [description]
        """
        raise NotImplementedError

    @property
    def const_hess(self):
        return False


class MeanSquaredError(GradLoss):

    def __init__(self, **kwargs):
        self.classification = False

    def loss(self, y, y_hat, *args):
        """The mean squared loss

        Args:
            y (Matrix): [n_instance, n_outcome]
            y_hat (Matrix): [n_instance, n_outcome] or [n_outcome]

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        return (y - y_hat)**2

    def fast_loss(self, y, y_hat, **kwargs):
        """\\sum((y- y_hat)^2) = \\sum y^2 + \\hat{y}^2-2\\hat{y}\\bar{y}

        Args:
            y (Matrix): [n_instance, n_outcome]
            y_hat (Matrix): [n_instance, n_outcome] or [n_outcome]
            reduction (_type_, optional): _description_. Defaults to sum.

        Returns:
            _type_: _description_
        """
        remove_squared = kwargs.pop("remove_squared", True)
        y_bar = kwargs.get('y_bar', None)
        y2_mean = kwargs.get('y2_mean', None)

        if y_bar is None:
            y_bar = np.mean(y, axis=0)
        if remove_squared is False:
            assert y2_mean is not None, f'you must supply `y2_mean` when remove_squared is False!'
        else:
            y2_mean = 0
        y_hat2_sum = (y_hat**2)
        cross = (2 * y_hat * y_bar)
        return y2_mean + y_hat2_sum - cross

    def approx_loss(self, Gs, **kwargs):
        """Approximate the loss by the second-order gradient

        Args:
            Gs (Matrix): [2, n_treatment, n_outcome]

        Returns:
            _type_: _description_
        """
        lambd = kwargs.get('lambd', 0)
        y_hat = kwargs.get('y_hat', None)
        G, H = Gs[0], Gs[1]
        if y_hat is None:
            return -.5 * (G**2) / (H + lambd)
        return G * y_hat + .5 * (y_hat**2) * (H + lambd)

    def gradients(self, target, prediction) -> Tuple:
        if isinstance(prediction, (int, float)):
            pred = np.full_like(target, prediction, target.dtype)
        else:
            pred = prediction
        # gradient
        gs = self.gradient(target, pred)
        hs = self.hessian(target, pred)

        return gs, hs

    def gradient(self, target, prediction):
        return (prediction - target) * 2

    def hessian(self, target, prediction):
        if np.isscalar(prediction):
            return np.full_like(target, 2, target.dtype)
        return np.full_like(prediction, 2, prediction.dtype)

    @property
    def const_hess(self):
        return True


class BinaryCrossEntropy(GradLoss):

    def __init__(self, **kwargs):
        self.classification = True

    def loss(self, target, prediction, logit=True):
        """calculate the cross entropy

        Args:
            target (Matrix): ground-truth label
            prediction (Matrix): prediction of logits
            logit (bool, optional): [description]. Defaults to True.
        """
        if logit:
            prob = sigmoid(prediction)
            ce = 0 - target * (prob + epsilon).apply(log) - (1 - target) * (1 - prob + epsilon).apply(log)
        else:
            ce = 0 - target * np.log(prediction + epsilon) - (1 - target) * np.log(1 - prediction + epsilon)
        return ce

    def fast_loss(self, y_hat, y_bar, reduction=sum, *args):
        """`loss_g(\mathbb{S};t<t_0) = \hat{y}^2-2\hat{y}\bar{y}`

        Args:
            y_hat (Matrix): The prediction of the model, [n_outcome]
            y_bar (Matrix): The average of current subset, [n_outcome]
        """
        # TODO
        raise NotImplementedError

    def approx_loss(self, Gs, **kwargs):
        """Approximate the loss by the second-order gradient

        Args:
            Gs (Matrix): [2, n_treatment, n_outcome]

        Returns:
            _type_: _description_
        """
        lambd = kwargs.get('lambd', 0)
        G, H = Gs[0], Gs[1]
        return -.5 * (G**2) / (H + lambd)

    def gradients(self, target, logit, treatment):
        """calculate gradient and hessian

        Args:
            target (DataFrame): [description]
            prediction (DataFrame): [description]
            treatment (DataFrame): [description]

        Returns:
            Union[Tuple, None]: [description]
        """
        if isinstance(logit, (int, float)):
            logit = pd.DataFrame(logit, index=target.index, columns=target.columns)
        probability = sigmoid(logit)
        # gradient
        gs = self.gradient(target, probability)
        hs = self.hessian(target, probability)

        return gs, hs

    def gradient(self, target, prediction):
        """```gradient = p-y``` p is the postive probability

        Args:
            target Matrix: [description]
            prediction Matrix: probability of prediction

        Returns:
            [type]: [description]
        """
        return prediction - target

    def hessian(self, target, prediction):
        """```hessian = p*(1-p)```

        Args:
            target (Matrix): [description]
            prediction (Matrix): probability of prediction

        Returns:
            [type]: [description]
        """
        return (prediction * (1 - prediction))[target.notna()]

    @property
    def const_hess(self):
        return False

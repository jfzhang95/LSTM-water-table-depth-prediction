#!usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from collections import OrderedDict


def floatX(X):
    """ Change data to theano type """
    return np.asarray(X, dtype=theano.config.floatX)


def clip(X, epsilon):
    """ Clip gradients """
    return T.maximum(T.minimum(X, epsilon), -1*epsilon)


def scale(X, max_norm):
    """ Gradients norm scale"""
    curr_norm = T.sum(T.abs_(X))
    return ifelse(T.lt(curr_norm, max_norm), X, max_norm * (X / curr_norm))


def get_params(layers):
    params = []
    for layer in layers:
        for param in layer.get_params():
            params.append(param)
    return params


def sgd(loss, params, learning_rate, clip_at=5.0, scale_norm=5.0):
    """ Stochastic Gradient Descent"""
    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)
    epsilon = 1e-8

    for p, grad in zip(params, grads):
        # if clip_at > 0.0:
        #     grad = clip(grad, clip_at)
        #
        # if scale_norm > 0.0:
        #     grad = scale(grad, scale_norm)
        grad_norm = grad.norm(L=2)
        grad = (T.minimum(clip_at, grad_norm) / (grad_norm + epsilon)) * grad
        updates[p] = p - learning_rate * grad

    return updates, grads


def adam(loss, params, learning_rate=1e-1, clip_at=5.0, scale_norm=5.0):
    updates = OrderedDict()
    grads = T.grad(loss, params)

    beta1 = 0.9
    beta2 = 0.995
    epsilon = 1e-8

    for p, grad in zip(params, grads):
        c = theano.shared(np.zeros_like(p.get_value(borrow=True)))
        # if clip_at > 0.0:
        #     grad = clip(g, clip_at)
        # else:
        #     grad = g
        #
        # if scale_norm > 0.0:
        #     grad = scale(grad, scale_norm)

        grad_norm = grad.norm(L=2)
        grad = (T.minimum(clip_at, grad_norm) / (grad_norm + epsilon)) * grad

        m = c
        v = c
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        updates[p] = p - learning_rate * m / (T.sqrt(v) + epsilon)

    return updates, grads


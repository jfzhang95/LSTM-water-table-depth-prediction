#!usr/bin/env python
#-*- coding:utf-8 -*-


import theano
import theano.tensor as T
from collections import OrderedDict
from utils import clip, scale
import numpy as np


def sgd(loss, params, learning_rate, clip_at=5.0, scale_norm=0.0):

    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)
    epsilon = 1e-8

    for p, g in zip(params, grads):
        # if clip_at > 0.0:
        #     grad = clip(g, clip_at)
        # else:
        #     grad = g
        #
        # if scale_norm > 0.0:
        #     grad = scale(grad, scale_norm)
        grad_norm = g.norm(L=2)
        grad = (T.minimum(clip_at, grad_norm) / (grad_norm + epsilon)) * g

        updates[p] = p - learning_rate * grad
    return updates, grads


def sgd_momentum(loss, params, learning_rate=1e-1, clip_at=5.0, scale_norm=0.0):
    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)


    momentum=0.9

    for p, g in zip(params, grads):
        c = theano.shared(np.zeros_like(p.get_value(borrow=True)))
        if clip_at > 0.0:
            grad = clip(g, clip_at)
        else:
            grad = g

        if scale_norm > 0.0:
            grad = scale(grad, scale_norm)

        v = momentum * c - learning_rate * grad
        updates[p] = p + v

    return updates, grads


def adagrad(loss, params, learning_rate, clip_at=5.0, scale_norm=0.0):
    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)

    epsilon = 1e-8

    for p, g in zip(params, grads):
        c = theano.shared(np.zeros_like(p.get_value(borrow=True)))
        if clip_at > 0.0:
            grad = clip(g, clip_at)
        else:
            grad = g

        if scale_norm > 0.0:
            grad = scale(grad, scale_norm)

    c += grad ** 2
    updates[p] = p - learning_rate * grad / (T.sqrt(c) + epsilon)
    return updates, grads


def adadelta(loss, params, learning_rate, clip_at=5.0, scale_norm=0.0):
    
    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)


    epsilon = 1e-8
    rho = learning_rate

    for p, g in zip(params, grads):
        c = theano.shared(np.zeros_like(p.get_value(borrow=True)))
        if clip_at > 0.0:
            grad = clip(g, clip_at)
        else:
            grad = g

        if scale_norm > 0.0:
            grad = scale(grad, scale_norm)

        gsum = c
        xsum = c
        gsum = rho * gsum + (1 - rho) * (grad ** 2)
        dpram = -T.sqrt((xsum + epsilon) / (gsum + epsilon)) * grad
        xsum = rho * xsum + (1 - rho) * (dpram ** 2)

        updates[p] = p + dpram

    return updates, grads


def rmsprop(loss, params, learning_rate=1e-2, clip_at=5.0, scale_norm=0.0):
    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)

    epsilon = 1e-8
    decay_rate = 0.90

    for p, g in zip(params, grads):
        c = theano.shared(np.zeros_like(p.get_value(borrow=True)))
        if clip_at > 0.0:
            grad = clip(g, clip_at)
        else:
            grad = g

        if scale_norm > 0.0:
            grad = scale(grad, scale_norm)

        c = decay_rate * c + (1 - decay_rate) * grad ** 2
        updates[p] = p - learning_rate * grad / (T.sqrt(c) + epsilon)

    return updates, grads


def adam(loss, params, learning_rate=1e-1, clip_at=5.0, max_norm=5.0):
    updates = OrderedDict()
    grads = T.grad(loss, params)


    beta1 = 0.9
    beta2 = 0.995
    epsilon = 1e-8

    for p, g in zip(params, grads):
        c = theano.shared(np.zeros_like(p.get_value(borrow=True)))
        # if clip_at > 0.0:
        #     grad = clip(g, clip_at)
        # else:
        #     grad = g
        #
        # if scale_norm > 0.0:
        #     grad = scale(grad, scale_norm)

        grad_norm = g.norm(L=2)
        grad = (T.minimum(max_norm, grad_norm) / (grad_norm + epsilon)) * g


        m = c
        v = c
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        updates[p] = p - learning_rate * m / (T.sqrt(v) + epsilon)

    return updates, grads









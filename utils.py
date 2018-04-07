#!usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def random_weights(shape, name=None):
    # return theano.shared(floatX(np.random.randn(*shape) * 0.01), name=name)
    return theano.shared(floatX(np.random.uniform(size=shape, low=-1, high=1)), name=name)


def zeros(shape, name=""):
    return theano.shared(floatX(np.zeros(shape)), name=name)


def dropout(X, dropout_prob=0.0):
    if dropout_prob < 0. or dropout_prob > 1.:
        raise Exception('Dropout level must be in interval [0, 1]')
    retain_prob = 1 - dropout_prob
    srng = RandomStreams(seed=1234)
    X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    X /= retain_prob
    return X


def clip(X, epsilon):
    return T.maximum(T.minimum(X, epsilon), -1*epsilon)


def scale(X, max_norm):
    curr_norm = T.sum(T.abs_(X))
    return ifelse(T.lt(curr_norm, max_norm), X, max_norm * (X / curr_norm))


def SGD(loss, params, learning_rate, lambda2=0.05):
    # problem in update
    # updates = {}
    # grads have no value??
    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)

    for p, g in zip(params, grads):
        # updates.append([p, p-learning_rate*(g+lambda2*p)])  # lambda*p regulzation
        updates[p] = p - learning_rate * (g + lambda2 * p)
    return updates, grads



def momentum(loss, params, caches, learning_rate=0.1, rho=0.1, clip_at=0.0, scale_norm=0.0, lambda2=0.0):
    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)

    for p, c, g in zip(params, caches, grads):
        if clip_at > 0.0:
            grad = clip(g, clip_at)
        else:
            grad = g

        if scale_norm > 0.0:
            grad = scale(grad, scale_norm)

        delta = rho * grad + (1-rho) * c
        updates[p] = p - learning_rate * (delta + lambda2 * p)

    return updates, grads




def get_params(layers):
    params = []
    for layer in layers:
        for param in layer.get_params():
            params.append(param)
    return params



def make_caches(params):
    caches = []
    for p in params:
        caches.append(theano.shared(floatX(np.zeros(p.get_value().shape))))
    return caches


def one_step_updates(layers):
    updates = []

    for layer in layers:
        updates += layer.updates()

    return updates




def sgd(loss, params, learning_rate, clip_at=0.0, scale_norm=0.0):
    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)
    epsilon = 1e-8

    for p, g in zip(params, grads):
        if clip_at > 0.0:
            grad = clip(g, clip_at)
        else:
            grad = g

        if scale_norm > 0.0:
            grad = scale(grad, scale_norm)
        grad_norm = grad.norm(L=2)
        grad = (T.minimum(clip_at, grad_norm) / (grad_norm + epsilon)) * grad
        updates[p] = p - learning_rate * grad

    return updates, grads


def sgd_momentum(loss, params, learning_rate=1e-1, clip_at=5.0, scale_norm=0.0):
    updates = OrderedDict()
    grads = T.grad(cost=loss, wrt=params)

    momentum = 0.9

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









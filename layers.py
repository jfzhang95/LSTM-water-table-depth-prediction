#!usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from utils import floatX


class NNLayer(object):
    def __init__(self):
        self.params = []

    def get_params_names(self):
        return ['UNK' if p.name is None else p.name for p in self.params]

    def save_model(self):
        return

    def load_model(self):
        return

    def updates(self):
        return

    def reset_state(self):
        return


class LSTMLayer(NNLayer):

    def __init__(self, num_input, num_hidden, input_layers=None, name="lstm"):
        """
        LSTM layer

        Arguments:
            num_input: previous layer's size
            num_hidden: hidden neurons' size
            input_layers: previous layer
        """
        self.name = name
        self.num_input = num_input
        self.num_hidden = num_hidden

        if len(input_layers) >= 2:
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()

        self.h0 = theano.shared(floatX(np.zeros(num_hidden)))
        self.s0 = theano.shared(floatX(np.zeros(num_hidden)))

        self.W_gx = self._random_weights((num_input, num_hidden), name=self.name+"W_gx")
        self.W_ix = self._random_weights((num_input, num_hidden), name=self.name+"W_ix")
        self.W_fx = self._random_weights((num_input, num_hidden), name=self.name+"W_fx")
        self.W_ox = self._random_weights((num_input, num_hidden), name=self.name+"W_ox")

        self.W_gh = self._random_weights((num_hidden, num_hidden), name=self.name+"W_gh")
        self.W_ih = self._random_weights((num_hidden, num_hidden), name=self.name+"W_ih")
        self.W_fh = self._random_weights((num_hidden, num_hidden), name=self.name+"W_fh")
        self.W_oh = self._random_weights((num_hidden, num_hidden), name=self.name+"W_oh")

        self.b_g = self._zeros(num_hidden, name=self.name+"b_g")
        self.b_i = self._zeros(num_hidden, name=self.name+"b_i")
        self.b_f = self._zeros(num_hidden, name=self.name+"b_f")
        self.b_o = self._zeros(num_hidden, name=self.name+"b_o")

        self.params = [self.W_gx, self.W_ix, self.W_ox, self.W_fx,
                       self.W_gh, self.W_ih, self.W_oh, self.W_fh,
                       self.b_g, self.b_i, self.b_f, self.b_o,
                       ]

        self.output()

    def _random_weights(self, shape, name=None):
        # return theano.shared(floatX(np.random.randn(*shape) * 0.01), name=name)
        return theano.shared(floatX(np.random.uniform(size=shape, low=-1, high=1)), name=name)

    def _zeros(self, shape, name=""):
        return theano.shared(floatX(np.zeros(shape)), name=name)

    def get_params(self):
        return self.params

    def _one_step(self, x, h_tm1, s_tm1):
        """
        Run the forward pass for a single time step of a LSTM layer

        Arguments:
            h_tm1: initial h
            s_tm1: initial s (cell state)

        Returns:
            h and s after one forward step
        """
        g = T.tanh(T.dot(x, self.W_gx) + T.dot(h_tm1, self.W_gh) + self.b_g)
        i = T.nnet.sigmoid(T.dot(x, self.W_ix) + T.dot(h_tm1, self.W_ih) + self.b_i)
        f = T.nnet.sigmoid(T.dot(x, self.W_fx) + T.dot(h_tm1, self.W_fh) + self.b_f)
        o = T.nnet.sigmoid(T.dot(x, self.W_ox) + T.dot(h_tm1, self.W_oh) + self.b_o)

        s = i * g + s_tm1 * f
        h = T.tanh(s) * o
        return h, s


    def output(self, go_backwards=False):

        outputs_info = [self.h0, self.s0]

        ([outputs, _], updates) = theano.scan(
            fn=self._one_step,
            sequences=self.X,
            outputs_info = outputs_info,
            go_backwards=go_backwards
        )
        return outputs


    def _reset_state(self):
        self.h0 = theano.shared(floatX(np.zeros(self.num_hidden)))
        self.s0 = theano.shared(floatX(np.zeros(self.num_hidden)))


class FullyConnectedLayer(NNLayer):
    """
    Fully-connected layer
    """
    def __init__(self, num_input, num_output, input_layers, name=""):

        if len(input_layers) >= 2:
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()
        self.W_yh = self._random_weights((num_input, num_output),name="W_yh_FC")
        self.b_y = self._zeros(num_output, name="b_y_FC")
        self.params = [self.W_yh, self.b_y]

    def _random_weights(self, shape, name=None):
        # return theano.shared(floatX(np.random.randn(*shape) * 0.01), name=name)
        return theano.shared(floatX(np.random.uniform(size=shape, low=-1, high=1)), name=name)

    def _zeros(self, shape, name=""):
        return theano.shared(floatX(np.zeros(shape)), name=name)

    def output(self):
        return T.dot(self.X, self.W_yh) + self.b_y

    def get_params(self):
        return self.params


class InputLayer(NNLayer):
    """
    Input layer
    """
    def __init__(self, X, name=""):
        self.name = name
        self.X = X
        self.params = []

    def get_params(self):
        return self.params

    def output(self):
        return self.X


class DropoutLayer(NNLayer):
    """
    Dropout layer
    """
    def __init__(self, input_layer, dropout_prob=0.5, name="dropout"):
        self.name = name
        self.X = input_layer.output()
        self.params = []
        self.dropout_prob = dropout_prob

    def get_params(self):
        return self.params

    def output(self):
        return self._dropout(self.X, self.dropout_prob)

    def _dropout(self, X, dropout_prob=0.0):
        retain_prob = 1 - dropout_prob
        srng = RandomStreams(seed=1234)
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        if dropout_prob != 0.0:
            X *= retain_prob
        return X

#!usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from utils import dropout, floatX, zeros, random_weights


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

    def __init__(self, num_input, num_hidden, input_layers=None, name="lstm", go_backwards=False):
        """
        Define LSTM layer

        Arguments:
            num_input:
            num_hidden:
            go_backwards: a flag indicating if ``scan`` should go
        backwards through the sequences.
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

        self.go_backwards = go_backwards

        self.W_gx = random_weights((num_input, num_hidden), name=self.name+"W_gx")
        self.W_ix = random_weights((num_input, num_hidden), name=self.name+"W_ix")
        self.W_fx = random_weights((num_input, num_hidden), name=self.name+"W_fx")
        self.W_ox = random_weights((num_input, num_hidden), name=self.name+"W_ox")

        self.W_gh = random_weights((num_hidden, num_hidden), name=self.name+"W_gh")
        self.W_ih = random_weights((num_hidden, num_hidden), name=self.name+"W_ih")
        self.W_fh = random_weights((num_hidden, num_hidden), name=self.name+"W_fh")
        self.W_oh = random_weights((num_hidden, num_hidden), name=self.name+"W_oh")

        self.b_g = zeros(num_hidden, name=self.name+"b_g")
        self.b_i = zeros(num_hidden, name=self.name+"b_i")
        self.b_f = zeros(num_hidden, name=self.name+"b_f")
        self.b_o = zeros(num_hidden, name=self.name+"b_o")

        self.params = [self.W_gx, self.W_ix, self.W_ox, self.W_fx,
                       self.W_gh, self.W_ih, self.W_oh, self.W_fh,
                       self.b_g, self.b_i, self.b_f, self.b_o,
                       ]

        self.output()

    def get_params(self):
        return self.params

    def one_step(self, x, h_tm1, s_tm1):
        """
        Run the forward pass for a single time step of a LSTM layer

        Arguments:
            h_tm1: initial h
            s_tm1: initial s (cell state)

        Returns:
            h and s after one forward pass step
        """
        g = T.tanh(T.dot(x, self.W_gx) + T.dot(h_tm1, self.W_gh) + self.b_g)
        i = T.nnet.sigmoid(T.dot(x, self.W_ix) + T.dot(h_tm1, self.W_ih) + self.b_i)
        f = T.nnet.sigmoid(T.dot(x, self.W_fx) + T.dot(h_tm1, self.W_fh) + self.b_f)
        o = T.nnet.sigmoid(T.dot(x, self.W_ox) + T.dot(h_tm1, self.W_oh) + self.b_o)

        s = i * g + s_tm1 * f
        h = T.tanh(s) * o
        return h, s


    def output(self, train=True):

        outputs_info = [self.h0, self.s0]

        ([outputs, states], updates) = theano.scan(
            fn=self.one_step,
            sequences=self.X,
            outputs_info = outputs_info,
            go_backwards=self.go_backwards
        )
        return outputs


    def reset_state(self):
        self.h0 = theano.shared(floatX(np.zeros(self.num_hidden)))
        self.s0 = theano.shared(floatX(np.zeros(self.num_hidden)))


class FullyConnectedLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, input_layers, name=""):

        if len(input_layers) >= 2:
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()
        self.W_yh = random_weights((num_input, num_output),name="W_yh_FC")
        self.b_y = zeros(num_output, name="b_y_FC")
        self.params = [self.W_yh, self.b_y]

    def output(self):
        # return T.nnet.sigmoid(T.dot(self.X, self.W_yh) + self.b_y)
        return T.dot(self.X, self.W_yh) + self.b_y

    def get_params(self):
        return self.params


class InputLayer(NNLayer):
    """
    """
    def __init__(self, X, name=""):
        self.name = name
        self.X = X
        self.params = []

    def get_params(self):
        return self.params

    def output(self, train=False):
        return self.X


class DropoutLayer(NNLayer):
    """
    """
    def __init__(self, input_layer, name="dropout", dropout_prob=0.5):
        self.X = input_layer.output()
        self.params = []
        self.dropout_prob = dropout_prob

    def get_params(self):
        return self.params

    def output(self):
        return dropout(self.X, self.dropout_prob)

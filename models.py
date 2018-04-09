#!usr/bin/env python
#-*- coding:utf-8 -*-

from layers import *
from utils import *
import gzip
import pickle

#########################################
#       LSTM-FC Model architecture      #
#########################################

class LSTM_FC_Model:

    def __init__(self, num_input=5, num_hidden=[64, 64], num_output=1, clip_at=0.0, scale_norm=0.0):
        """
        LSTM-FC Model, lstm layer contributes to learning time series data, dropout helps to prevent overfitting.

        Arguments:
            num_input: the number of input variables
            num_hidden: the number of neurons in each hidden layer
            num_output: output size (one in this study)
            clip_at: gradient clip
            scale_norm: gradient norm scale

        Returns:
            output (water table depth in this study)
        """
        print('Build LSTM_FC Model......')

        X = T.fmatrix()
        Y = T.fmatrix()
        learning_rate = T.fscalar()
        dropout_prob = T.fscalar()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.clip_at = clip_at
        self.scale_norm = scale_norm

        inputs = InputLayer(X, name='inputs')
        num_prev = num_input
        prev_layer = inputs

        self.layers = [inputs]

        for i, num_curr in enumerate(num_hidden):
            lstm = LSTMLayer(num_prev, num_curr, input_layers=[prev_layer], name="lstm{0}".format(i + 1))

            num_prev = num_curr
            prev_layer = lstm
            self.layers.append(prev_layer)
            prev_layer = DropoutLayer(prev_layer, dropout_prob)
            self.layers.append(prev_layer)

        fc = FullyConnectedLayer(num_prev, num_output, input_layers=[prev_layer], name="yhat")
        self.layers.append(fc)
        Y_hat = fc.output()

        loss = T.sum((Y - Y_hat) ** 2)
        params = get_params(self.layers)

        # You can also use adam optimization method
        updates, grads = sgd(loss, params, learning_rate)
        # updates, grads = adam(loss, params, learning_rate)


        self.train_func = theano.function([X, Y, learning_rate, dropout_prob], loss, updates=updates, allow_input_downcast=True)

        self.predict_func = theano.function([X, dropout_prob], Y_hat, allow_input_downcast=True)


    def fit(self, X, Y, learning_rate, dropout_prob):
        return self.train_func(X, Y, learning_rate, dropout_prob)


    def predict(self, X):
        return self.predict_func(X, 0.0)


    def save_model_params(self, filename):
        to_save = {'num_input': self.num_input, 'num_hidden': self.num_hidden,
                   'num_output': self.num_output}

        for layer in self.layers:
            for p in layer.get_params():
                assert (p.name not in to_save)
                to_save[p.name] = p.get_value()

        with gzip.open(filename, 'wb') as f:
            pickle.dump(to_save, f)


    def load_model_params(self, filename):
        f = gzip.open(filename, 'rb')
        to_load = pickle.load(f)
        assert (to_load['num_input'] == self.num_input)
        assert (to_load['num_output'] == self.num_output)

        for layer in self.layers:
            for p in layer.get_params():
                p.set_value(floatX(to_load[p.name]))



##########################################
#         FFNN Model architecture        #
##########################################

class FFNN_Model:

    def __init__(self, num_input=256, num_hidden=[64,64], num_output=1, clip_at=0.0, scale_norm=0.0):
        """
        FFNN Model, two hidden fully-connected layers.

        Arguments:
            num_input: the number of input variables
            num_hidden: the number of neurons in each hidden layer
            num_output: output size (one in this study)
            clip_at: gradient clip
            scale_norm: gradient norm scale

        Returns:
            output (water table depth in this study)
        """
        print('Build FFNN Model......')

        X = T.fmatrix()
        Y = T.fmatrix()
        learning_rate = T.fscalar()
        dropout_prob = T.fscalar()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.clip_at = clip_at
        self.scale_norm = scale_norm

        inputs = InputLayer(X, name='inputs')
        num_prev = num_input
        prev_layer = inputs

        self.layers = [inputs]
        fc = FullyConnectedLayer(num_prev, num_hidden, input_layers=[prev_layer], name="fc")
        num_prev = num_hidden
        prev_layer = fc
        self.layers.append(prev_layer)
        prev_layer = DropoutLayer(prev_layer, dropout_prob)
        self.layers.append(prev_layer)

        fc = FullyConnectedLayer(num_prev, num_output, input_layers=[prev_layer], name="yhat")
        self.layers.append(fc)
        Y_hat = fc.output()

        loss = T.sum((Y - Y_hat) ** 2)
        params = get_params(self.layers)

        updates, grads = sgd(loss, params, learning_rate)


        self.train_func = theano.function([X, Y, learning_rate, dropout_prob], loss, updates=updates, allow_input_downcast=True)

        self.predict_func = theano.function([X, dropout_prob], Y_hat, allow_input_downcast=True)


    def fit(self, X, Y, learning_rate, dropout_prob):
        return self.train_func(X, Y, learning_rate, dropout_prob)


    def predict(self, X):
        return self.predict_func(X, 0.0)  # in predict time, dropout = 0


    def save_model_params(self, filename):
        to_save = {'num_input': self.num_input, 'num_hidden': self.num_hidden,
                   'num_output': self.num_output}

        for layer in self.layers:
            for p in layer.get_params():
                assert (p.name not in to_save)
                to_save[p.name] = p.get_value()

        with gzip.open(filename, 'wb') as f:
            pickle.dump(to_save, f)


    def load_model_params(self, filename):
        f = gzip.open(filename, 'rb')
        to_load = pickle.load(f)
        assert (to_load['num_input'] == self.num_input)
        assert (to_load['num_output'] == self.num_output)

        for layer in self.layers:
            for p in layer.get_params():
                p.set_value(floatX(to_load[p.name]))


#########################################
#     Double-LSTM Model architecture    #
#########################################

class Double_LSTM_Model:
    """
    Double-LSTM Model, two hidden lstm layers.

    Arguments:
        num_input: the number of input variables
        num_hidden: the number of neurons in each hidden layer
        num_output: output size (one in this study)
        clip_at: gradient clip
        scale_norm: gradient norm scale

    Returns:
        output (water table depth in this study)
    """
    def __init__(self, num_input=256, num_hidden=[64,64], num_output=1, clip_at=0.0, scale_norm=0.0):
        print('Build Double_LSTM Model......')

        X = T.fmatrix()
        Y = T.fmatrix()
        learning_rate = T.fscalar()
        dropout_prob = T.fscalar()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.clip_at = clip_at
        self.scale_norm = scale_norm

        inputs = InputLayer(X, name='inputs')

        self.layers = [inputs]
        lstm = LSTMLayer(num_input, num_hidden, input_layers=[inputs], name="lstm{0}".format(1))
        self.layers.append(lstm)
        lstm_dropout = DropoutLayer(lstm, dropout_prob)
        self.layers.append(lstm_dropout)

        lstm = LSTMLayer(num_hidden, num_output, input_layers=[lstm_dropout], name="lstm{0}".format(2))
        self.layers.append(lstm)
            
        Y_hat = lstm.output()

        loss = T.sum((Y - Y_hat) ** 2)
        params = get_params(self.layers)

        updates, grads = sgd(loss, params, learning_rate, self.clip_at, self.scale_norm)

        self.train_func = theano.function([X, Y, learning_rate, dropout_prob], loss, updates=updates,
                                          allow_input_downcast=True)

        self.predict_func = theano.function([X, dropout_prob], Y_hat, allow_input_downcast=True)

    def fit(self, X, Y, learning_rate, dropout_prob):
        return self.train_func(X, Y, learning_rate, dropout_prob)


    def predict(self, X):
        return self.predict_func(X, 0.0)  # in predict time, dropout = 0


    def save_model_params(self, filename):
        to_save = {'num_input': self.num_input, 'num_hidden': self.num_hidden,
                   'num_output': self.num_output}

        for layer in self.layers:
            for p in layer.get_params():
                assert (p.name not in to_save)
                to_save[p.name] = p.get_value()

        with gzip.open(filename, 'wb') as f:
            pickle.dump(to_save, f)


    def load_model_params(self, filename):
        f = gzip.open(filename, 'rb')
        to_load = pickle.load(f)
        assert (to_load['num_input'] == self.num_input)
        assert (to_load['num_output'] == self.num_output)

        for layer in self.layers:
            for p in layer.get_params():
                p.set_value(floatX(to_load[p.name]))
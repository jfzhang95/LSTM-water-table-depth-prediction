#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date: 2017-03-15
"""


from layers import *
from methods import get_params, floatX
import types
from optim import *

import gzip
import cPickle
import sys


##############################################
#    LSTM used for predict evaporation       #
##############################################


class LSTM4eva:

    def __init__(self, num_input=256, num_hidden=[64,64], num_output=500, clip_at=5.0, scale_norm=0.0):
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
        if type(num_hidden) is types.IntType:
            lstm = LSTMLayer(num_prev, num_hidden, input_layers=[prev_layer], name="lstm", go_backwards=False)
            num_prev = num_hidden
            prev_layer = lstm
            self.layers.append(prev_layer)
            prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
            self.layers.append(prev_layer)

        else:
            for i, num_curr in enumerate(num_hidden):
                lstm = LSTMLayer(num_prev, num_curr, input_layers=[prev_layer], name="lstm{0}".format(i + 1), go_backwards=False)

                num_prev = num_curr
                prev_layer = lstm
                self.layers.append(prev_layer)
                prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
                self.layers.append(prev_layer)

        FC = FullyConnectedLayer(num_prev, num_output, input_layers=[prev_layer], name="yhat")
        self.layers.append(FC)
        Y_hat = FC.output()



        loss = T.sum((Y - Y_hat) ** 2) + 0.5 * T.sum(FC.W_yh * FC.W_yh)

        params = get_params(self.layers)

        updates, grads = adam(loss, params, learning_rate)


        self.train_func = theano.function([X, Y, learning_rate, dropout_prob], loss, updates=updates, allow_input_downcast=True)

        self.predict_func = theano.function([X, dropout_prob], Y_hat, allow_input_downcast=True)


    def fit(self, X, Y, learning_rate, dropout_prob):
        return self.train_func(X, Y, learning_rate, dropout_prob)

    def predict(self, X):
        return self.predict_func(X, 0.0)  # in predict time dropout = 0

    def save_model_params(self, filename):
        to_save = {'num_input': self.num_input, 'num_hidden': self.num_hidden,
                   'num_output': self.num_output}

        for layer in self.layers:
            for p in layer.get_params():
                print(p.get_value)

                assert (p.name not in to_save)
                to_save[p.name] = p.get_value()

        with gzip.open(filename, 'wb') as f:
            cPickle.dump(to_save, f)

    def load_model_params(self, filename):
        f = gzip.open(filename, 'rb')
        to_load = cPickle.load(f)
        assert (to_load['num_input'] == self.num_input)
        assert (to_load['num_output'] == self.num_output)

        saved_num_hidden = to_load['num_hidden']

        # try:
        #     len(saved_num_hidden)
        # except:
        #     assert (np.all([saved_num_hidden == h for h in self.num_hidden]))
        # else:
        #     assert (len(saved_num_hidden) == len(self.num_hidden))
        #     assert (np.all([hi == h2 for hi, h2 in zip(saved_num_hidden, self.num_hidden)]))

        for layer in self.layers:
            for p in layer.get_params():
                p.set_value(floatX(to_load[p.name]))



##############################################
#    LSTM used for predict water_discharge   #
##############################################

class LSTM4wd:

    def __init__(self, num_input=256, num_hidden=[64,64], num_output=500, clip_at=5.0, scale_norm=0.0):
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
        if type(num_hidden) is types.IntType:
            lstm = LSTMLayer(num_prev, num_hidden, input_layers=[prev_layer], name="lstm", go_backwards=False)
            num_prev = num_hidden
            prev_layer = lstm
            self.layers.append(prev_layer)
            prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
            self.layers.append(prev_layer)

        else:
            for i, num_curr in enumerate(num_hidden):
                lstm = LSTMLayer(num_prev, num_curr, input_layers=[prev_layer], name="lstm{0}".format(i + 1), go_backwards=False)

                num_prev = num_curr
                prev_layer = lstm
                self.layers.append(prev_layer)
                prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
                self.layers.append(prev_layer)

        FC = FullyConnectedLayer(num_prev, num_output, input_layers=[prev_layer], name="yhat")
        self.layers.append(FC)
        Y_hat = FC.output()



        loss = T.sum((Y - Y_hat.T) ** 2) + 1 * T.sum(FC.W_yh * FC.W_yh)
        params = get_params(self.layers)

        updates, grads = adam(loss, params, learning_rate)


        self.train_func = theano.function([X, Y, learning_rate, dropout_prob], loss, updates=updates, allow_input_downcast=True)

        self.predict_func = theano.function([X, dropout_prob], Y_hat, allow_input_downcast=True)


    def fit(self, X, Y, learning_rate, dropout_prob):
        return self.train_func(X, Y, learning_rate, dropout_prob)

    def predict(self, X):
        return self.predict_func(X, 0.0)  # in predict time dropout = 0

    def save_model_params(self, filename):
        to_save = {'num_input': self.num_input, 'num_hidden': self.num_hidden,
                   'num_output': self.num_output}

        for layer in self.layers:
            for p in layer.get_params():
                assert (p.name not in to_save)
                to_save[p.name] = p.get_value()

        with gzip.open(filename, 'wb') as f:
            cPickle.dump(to_save, f)

    def load_model_params(self, filename):
        f = gzip.open(filename, 'rb')
        to_load = cPickle.load(f)
        assert (to_load['num_input'] == self.num_input)
        assert (to_load['num_output'] == self.num_output)

        saved_num_hidden = to_load['num_hidden']

        # try:
        #     len(saved_num_hidden)
        # except:
        #     assert (np.all([saved_num_hidden == h for h in self.num_hidden]))
        # else:
        #     assert (len(saved_num_hidden) == len(self.num_hidden))
        #     assert (np.all([hi == h2 for hi, h2 in zip(saved_num_hidden, self.num_hidden)]))

        for layer in self.layers:
            for p in layer.get_params():
                p.set_value(floatX(to_load[p.name]))



####################################
#    LSTM used for predict depth   #
####################################



class LSTM4dep:

    def __init__(self, num_input=256, num_hidden=[64,64], num_output=500, clip_at=5.0, scale_norm=0.0):
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
        if type(num_hidden) is types.IntType:
            lstm = LSTMLayer4wd(num_prev, num_hidden, input_layers=[prev_layer], name="lstm", go_backwards=False)
            num_prev = num_hidden
            prev_layer = lstm
            self.layers.append(prev_layer)
            prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
            self.layers.append(prev_layer)

        else:
            for i, num_curr in enumerate(num_hidden):
                lstm = LSTMLayer4wd(num_prev, num_curr, input_layers=[prev_layer], name="lstm{0}".format(i + 1), go_backwards=False)

                num_prev = num_curr
                prev_layer = lstm
                self.layers.append(prev_layer)
                prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
                self.layers.append(prev_layer)

        FC = FullyConnectedLayer(num_prev, num_output, input_layers=[prev_layer], name="yhat")
        self.layers.append(FC)
        Y_hat = FC.output()



        loss = T.sum((Y - Y_hat) ** 2) + 0.5 * T.sum(FC.W_yh * FC.W_yh)
        params = get_params(self.layers)

        updates, grads = adam(loss, params, learning_rate)


        self.train_func = theano.function([X, Y, learning_rate, dropout_prob], loss, updates=updates, allow_input_downcast=True)

        self.predict_func = theano.function([X, dropout_prob], Y_hat, allow_input_downcast=True)


    def fit(self, X, Y, learning_rate, dropout_prob):
        return self.train_func(X, Y, learning_rate, dropout_prob)

    def predict(self, X):
        return self.predict_func(X, 0.0)  # in predict time dropout = 0


    def save_model_params(self, filename):
        to_save = {'num_input': self.num_input, 'num_hidden': self.num_hidden,
                   'num_output': self.num_output}

        for layer in self.layers:
            for p in layer.get_params():
                assert (p.name not in to_save)
                to_save[p.name] = p.get_value()

        with gzip.open(filename, 'wb') as f:
            cPickle.dump(to_save, f)


    def load_model_params(self, filename):
        f = gzip.open(filename, 'rb')
        to_load = cPickle.load(f)
        assert (to_load['num_input'] == self.num_input)
        assert (to_load['num_output'] == self.num_output)

        saved_num_hidden = to_load['num_hidden']

        # try:
        #     len(saved_num_hidden)
        # except:
        #     assert (np.all([saved_num_hidden == h for h in self.num_hidden]))
        # else:
        #     assert (len(saved_num_hidden) == len(self.num_hidden))
        #     assert (np.all([hi == h2 for hi, h2 in zip(saved_num_hidden, self.num_hidden)]))

        for layer in self.layers:
            for p in layer.get_params():
                p.set_value(floatX(to_load[p.name]))


                
                
class ANN4dep:

    def __init__(self, num_input=256, num_hidden=[64,64], num_output=500, clip_at=5.0, scale_norm=0.0):
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
        if type(num_hidden) is types.IntType:
            ann = FullyConnectedLayer(num_prev, num_hidden, input_layers=[prev_layer], name="ann")
            num_prev = num_hidden
            prev_layer = ann
            self.layers.append(prev_layer)
            prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
            self.layers.append(prev_layer)

        else:
            for i, num_curr in enumerate(num_hidden):
                lstm = LSTMLayer4wd(num_prev, num_curr, input_layers=[prev_layer], name="lstm{0}".format(i + 1), go_backwards=False)

                num_prev = num_curr
                prev_layer = lstm
                self.layers.append(prev_layer)
                prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
                self.layers.append(prev_layer)

        FC = FullyConnectedLayer(num_prev, num_output, input_layers=[prev_layer], name="yhat")
        self.layers.append(FC)
        Y_hat = FC.output()



        loss = T.sum((Y - Y_hat) ** 2) + 0.5 * T.sum(FC.W_yh * FC.W_yh)
        params = get_params(self.layers)

        updates, grads = adam(loss, params, learning_rate)


        self.train_func = theano.function([X, Y, learning_rate, dropout_prob], loss, updates=updates, allow_input_downcast=True)

        self.predict_func = theano.function([X, dropout_prob], Y_hat, allow_input_downcast=True)


    def fit(self, X, Y, learning_rate, dropout_prob):
        return self.train_func(X, Y, learning_rate, dropout_prob)

    def predict(self, X):
        return self.predict_func(X, 0.0)  # in predict time dropout = 0


    def save_model_params(self, filename):
        to_save = {'num_input': self.num_input, 'num_hidden': self.num_hidden,
                   'num_output': self.num_output}

        for layer in self.layers:
            for p in layer.get_params():
                assert (p.name not in to_save)
                to_save[p.name] = p.get_value()

        with gzip.open(filename, 'wb') as f:
            cPickle.dump(to_save, f)


    def load_model_params(self, filename):
        f = gzip.open(filename, 'rb')
        to_load = cPickle.load(f)
        assert (to_load['num_input'] == self.num_input)
        assert (to_load['num_output'] == self.num_output)

        saved_num_hidden = to_load['num_hidden']

        # try:
        #     len(saved_num_hidden)
        # except:
        #     assert (np.all([saved_num_hidden == h for h in self.num_hidden]))
        # else:
        #     assert (len(saved_num_hidden) == len(self.num_hidden))
        #     assert (np.all([hi == h2 for hi, h2 in zip(saved_num_hidden, self.num_hidden)]))

        for layer in self.layers:
            for p in layer.get_params():
                p.set_value(floatX(to_load[p.name]))
                
                
                
                
                
class doubleLSTM4dep:

    def __init__(self, num_input=256, num_hidden=[64,64], num_output=500, clip_at=5.0, scale_norm=0.0):
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
        lstm = LSTMLayer4wd(num_prev, num_hidden, input_layers=[prev_layer], name="lstm{0}".format(1), go_backwards=False)
        prev_layer = lstm
        self.layers.append(prev_layer)
        prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
        self.layers.append(prev_layer)

        lstm = LSTMLayer4wd(num_hidden, num_output, input_layers=[prev_layer], name="lstm{0}".format(2), go_backwards=False)
        self.layers.append(lstm)
            
        Y_hat = lstm.output()


        print("test successed!")
        loss = T.sum((Y - Y_hat) ** 2)
        params = get_params(self.layers)

        updates, grads = adam(loss, params, learning_rate)


        self.train_func = theano.function([X, Y, learning_rate, dropout_prob], loss, updates=updates, allow_input_downcast=True)

        self.predict_func = theano.function([X, dropout_prob], Y_hat, allow_input_downcast=True)


    def fit(self, X, Y, learning_rate, dropout_prob):
        return self.train_func(X, Y, learning_rate, dropout_prob)

    def predict(self, X):
        return self.predict_func(X, 0.0)  # in predict time dropout = 0


    def save_model_params(self, filename):
        to_save = {'num_input': self.num_input, 'num_hidden': self.num_hidden,
                   'num_output': self.num_output}

        for layer in self.layers:
            for p in layer.get_params():
                assert (p.name not in to_save)
                to_save[p.name] = p.get_value()

        with gzip.open(filename, 'wb') as f:
            cPickle.dump(to_save, f)


    def load_model_params(self, filename):
        f = gzip.open(filename, 'rb')
        to_load = cPickle.load(f)
        assert (to_load['num_input'] == self.num_input)
        assert (to_load['num_output'] == self.num_output)

        saved_num_hidden = to_load['num_hidden']

        # try:
        #     len(saved_num_hidden)
        # except:
        #     assert (np.all([saved_num_hidden == h for h in self.num_hidden]))
        # else:
        #     assert (len(saved_num_hidden) == len(self.num_hidden))
        #     assert (np.all([hi == h2 for hi, h2 in zip(saved_num_hidden, self.num_hidden)]))

        for layer in self.layers:
            for p in layer.get_params():
                p.set_value(floatX(to_load[p.name]))

########################################
#    LSTM used for predict all depth   #
########################################
 
class LSTM4alleva:

    def __init__(self, num_input=256, num_hidden=[64,64], num_output=500, clip_at=5.0, scale_norm=0.0):
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
        if type(num_hidden) is types.IntType:
            lstm = LSTMLayer4wd(num_prev, num_hidden, input_layers=[prev_layer], name="lstm", go_backwards=False)
            num_prev = num_hidden
            prev_layer = lstm
            self.layers.append(prev_layer)
            prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
            self.layers.append(prev_layer)

        else:
            for i, num_curr in enumerate(num_hidden):
                lstm = LSTMLayer4wd(num_prev, num_curr, input_layers=[prev_layer], name="lstm{0}".format(i + 1), go_backwards=False)

                num_prev = num_curr
                prev_layer = lstm
                self.layers.append(prev_layer)
                prev_layer = DropoutLayer(prev_layer, dropout_prob=dropout_prob)
                self.layers.append(prev_layer)

        FC = FullyConnectedLayer(num_prev, num_output, input_layers=[prev_layer], name="fc")
        self.layers.append(FC)
        Y_hat = FC.output()



        loss = T.sum((Y - Y_hat) ** 2) + 0.5 * T.sum(FC.W_yh * FC.W_yh)
        params = get_params(self.layers)

        updates, grads = adam(loss, params, learning_rate)


        self.train_func = theano.function([X, Y, learning_rate, dropout_prob], loss, updates=updates, allow_input_downcast=True)

        self.predict_func = theano.function([X, dropout_prob], Y_hat, allow_input_downcast=True)


    def fit(self, X, Y, learning_rate, dropout_prob):
        return self.train_func(X, Y, learning_rate, dropout_prob)

    def predict(self, X):
        return self.predict_func(X, 0.0)  # in predict time dropout = 0


    def save_model_params(self, filename):
        to_save = {'num_input': self.num_input, 'num_hidden': self.num_hidden,
                   'num_output': self.num_output}

        for layer in self.layers:
            for p in layer.get_params():
                assert (p.name not in to_save)
                to_save[p.name] = p.get_value()

        with gzip.open(filename, 'wb') as f:
            cPickle.dump(to_save, f)


    def load_model_params(self, filename):
        f = gzip.open(filename, 'rb')
        to_load = cPickle.load(f)
        assert (to_load['num_input'] == self.num_input)
        assert (to_load['num_output'] == self.num_output)

        saved_num_hidden = to_load['num_hidden']

        # try:
        #     len(saved_num_hidden)
        # except:
        #     assert (np.all([saved_num_hidden == h for h in self.num_hidden]))
        # else:
        #     assert (len(saved_num_hidden) == len(self.num_hidden))
        #     assert (np.all([hi == h2 for hi, h2 in zip(saved_num_hidden, self.num_hidden)]))

        for layer in self.layers:
            for p in layer.get_params():
                p.set_value(floatX(to_load[p.name]))

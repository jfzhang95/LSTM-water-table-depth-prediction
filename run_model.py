#!usr/bin/env python
#-*- coding:utf-8 -*-


from rnn_model import *
import numpy as np

def LSTM_pred_dep(X, Y, X_test=None, iters=1000, learning_rate=1e-1, dropout_prob=0.5):
    print("lr:", learning_rate)
    print("dropout:", dropout_prob)
    print("iterations:", iters)
    num_count = Y.shape[0]
    input_shape = X.shape[1]
    print('num_count:', num_count)
    print('input_size:', input_shape)
    model = LSTM_FC_Model(num_input=input_shape, num_hidden=40, num_output=1)

    Loss = []
    for iter in range(iters+1):
        loss = model.fit(X, Y, learning_rate, dropout_prob)
        Loss.append(loss)
        if iter % 2000 == 0:
            print("iteration: %s, loss: %s" % (iter, loss))

    print('starting predicting......')
    Y_test = model.predict(X_test)
    print('predicting done!')
    return Y_test


def ANN_pred_dep(X, Y, X_test=None, iters=1000, learning_rate=1e-1, dropout_prob=0.5, index=0):
    #     print ("lr:", learning_rate)
    #     print ("dropout:", dropout_prob)
    #     print ("iterations:", iters)
    num_count = Y.shape[0]
    input_shape = X.shape[1]
    #     print ('num_count:', num_count)
    #     print ('input_size:', input_shape)
    # 加排水量的时候 hidden=34
    model = FFNN_Model(num_input=input_shape, num_hidden=80, num_output=1)

    Loss = []
    for iter in range(iters + 1):
        loss = model.fit(X, Y, learning_rate, dropout_prob)
        Loss.append(loss)
        if iter % 2000 == 0:
            print("[INFO] iteration: %s, loss: %s" % (iter, loss))

            #     rnn.save_model_params('../output/yichang/dep_param'+str(index))
    print('starting predicting......')
    Y_test = model.predict(X_test)
    #     print ('predicting done!')


    print('done!')
    return Y_test


def DoubleLSTM_pred_dep(X, Y, X_test=None, iters=1000, learning_rate=1e-1, dropout_prob=0.5):
    print("lr:", learning_rate)
    print("dropout:", dropout_prob)
    print("iterations:", iters)
    num_count = Y.shape[0]
    input_shape = X.shape[1]
    print('num_count:', num_count)
    print('input_size:', input_shape)
    model = Double_LSTM_Model(num_input=input_shape, num_hidden=40, num_output=1)

    Loss = []
    for iter in range(iters+1):
        loss = model.fit(X, Y, learning_rate, dropout_prob)
        Loss.append(loss)
        if iter % 2000 == 0:
            print("iteration: %s, loss: %s" % (iter, loss))

    print('starting predicting......')
    Y_test = model.predict(X_test)
    print('predicting done!')
    return Y_test


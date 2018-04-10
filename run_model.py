#!usr/bin/env python
#-*- coding:utf-8 -*-


from models import LSTM_FC_Model, FFNN_Model, Double_LSTM_Model


def LSTM_FC_prediction(X, Y, X_test=None, iters=20000, learning_rate=1e-4, dropout_prob=0.5):
    if dropout_prob > 1. or dropout_prob < 0.:
        raise Exception('Dropout level must be in interval [0, 1]')
    print("learning rate:", learning_rate)
    print("dropout:", dropout_prob)
    print("iterations:", iters)
    num_month = Y.shape[0]
    input_shape = X.shape[1]
    print('num_month:', num_month)
    print('variable size:', input_shape)
    model = LSTM_FC_Model(num_input=input_shape, num_hidden=[40], num_output=1)

    Loss = []
    print('Start training......')
    for iter in range(iters + 1):
        loss = model.fit(X, Y, learning_rate, dropout_prob)
        Loss.append(loss)
        if iter % 1000 == 0:
            print("iteration: %s, loss: %s" % (iter, loss))

    # Saving model
    model.save_model_params('checkpoints/LSTM_FC_CKPT')

    print('Start predicting......')
    Y_test = model.predict(X_test)
    print('Done.')
    return Y_test


def FFNN_prediction(X, Y, X_test=None, iters=20000, learning_rate=1e-4, dropout_prob=0.5):
    if dropout_prob > 1. or dropout_prob < 0.:
        raise Exception('Dropout level must be in interval [0, 1]')
    print ("learning_rate:", learning_rate)
    print ("dropout:", dropout_prob)
    print ("iterations:", iters)
    num_month = Y.shape[0]
    input_shape = X.shape[1]
    print('num_month:', num_month)
    print('variable size:', input_shape)
    model = FFNN_Model(num_input=input_shape, num_hidden=[40], num_output=1)

    Loss = []
    print('Start training......')
    for iter in range(iters + 1):
        loss = model.fit(X, Y, learning_rate, dropout_prob)
        Loss.append(loss)
        if iter % 1000 == 0:
            print("iteration: %s, loss: %s" % (iter, loss))

    model.save_model_params('checkpoints/FFNN_CKPT')

    print('Start predicting......')
    Y_test = model.predict(X_test)
    print('Done.')
    return Y_test


def DoubleLSTM_prediction(X, Y, X_test=None, iters=1000, learning_rate=1e-1, dropout_prob=0.5):
    if dropout_prob > 1. or dropout_prob < 0.:
        raise Exception('Dropout level must be in interval [0, 1]')
    print("learning_rate:", learning_rate)
    print("dropout:", dropout_prob)
    print("iterations:", iters)
    num_month = Y.shape[0]
    input_shape = X.shape[1]
    print('num_month:', num_month)
    print('variable size:', input_shape)
    model = Double_LSTM_Model(num_input=input_shape, num_hidden=[40], num_output=1)

    Loss = []
    print('Start training......')
    for iter in range(iters + 1):
        loss = model.fit(X, Y, learning_rate, dropout_prob)
        Loss.append(loss)
        if iter % 1000 == 0:
            print("iteration: %s, loss: %s" % (iter, loss))

    model.save_model_params('checkpoints/Double_LSTM_CKPT')

    print('Start predicting......')
    Y_test = model.predict(X_test)
    print('Done.')
    return Y_test


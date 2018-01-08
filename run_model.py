#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date: 2017-03-15
"""




import numpy as np
import csv
from rnn_model import *





# def LSTM_model_test(X, Y, X_test=None, y_test=None, iters=1000, learning_rate=1e-1, dropout_prob=0.5):
#     print ("lr:", learning_rate)
#     print ("dropout:", dropout_prob)
#     print ("iterations:", iters)
#     num_count = Y.shape[0]
#     input_shape = X.shape[1]
#     print ('num_count:', num_count)
#     print ('input_size:', input_shape)
#     rnn = LSTM(num_input=input_shape, num_hidden=18, num_output=1)
#
#     Loss = []
#     for iter in xrange(iters):
#         loss = rnn.train(X, Y, learning_rate, dropout_prob)
#         Loss.append(loss)
#         if iter % 1000 == 0:
#             print ("iteration: %s, loss: %s" % (iter, loss))
#
#     print ('starting predicting......')
#     Y_test = rnn.predict(X_test)
#     print ('predicting done!')
#     from LoadDataSet import ss_y
#     Y_test = ss_y.inverse_transform(Y_test)
#     y_test = ss_y.inverse_transform(y_test)
#
#
#
#     from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#     print ('the value of R-squared of DNN is ', r2_score(y_test, Y_test))
#     print ('the MSE of DNN is ', mean_squared_error(y_test, Y_test))
#     print ('the MAE of DNN is ', mean_absolute_error(y_test, Y_test))
#
#     print ('writing result to csv')
#     with open('../output/lstm_test.csv', 'wb') as MyFile:
#         myWriter = csv.writer(MyFile)
#         myWriter.writerow(["true", "  ", "pred"])
#         for true, pred in zip(y_test, Y_test):
#             tmp = []
#             tmp.append(np.float(true))
#             tmp.append('  ')
#             tmp.append(pred[0])
#             myWriter.writerow(tmp)
#
#     print ('done!')
#     import matplotlib.pyplot as plt
#
#
#
#     f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8,12))
#
#     ax1.plot(y_test, color='red')
#     ax1.plot(Y_test, color='green')
#     ax1.set_title('V.S.')
#     plt.xlabel('Month')
#     plt.ylabel('Depth')
#     plt.xlim(0, 25)
#
#
#     ax2.plot(Loss)
#     ax2.set_title('Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.xlim(0, 15000)
#     plt.ylim(0, 200)
#     plt.show()

    



def LSTM_pred_eva(X, Y, X_test=None, iters=1000, learning_rate=1e-1, dropout_prob=0.5):
    print ("lr:", learning_rate)
    print ("dropout:", dropout_prob)
    print ("iterations:", iters)
    num_count = Y.shape[0]
    input_shape = X.shape[1]
    print ('num_count:', num_count)
    print ('input_size:', input_shape)
    rnn = LSTM4eva(num_input=input_shape, num_hidden=20, num_output=1)

    Loss = []
    for iter in xrange(iters+1):
        loss = rnn.fit(X, Y, learning_rate, dropout_prob)
        Loss.append(loss)
        if iter % 2000 == 0:
            print ("iteration: %s, loss: %s" % (iter, loss))

    rnn.save_model_params('rnn_params')

    if X_test is not None:

        print ('starting predicting......')
        Y_test = rnn.predict(X_test)
        print ('predicting done!')


        print ('done!')
        return Y_test

        # print ('writing result to csv')
        # with open('output/pred_eva_data.csv', 'wb') as MyFile:
        #     myWriter = csv.writer(MyFile)
        #     myWriter.writerow(["Evaluation"])
        #     for pred in Y_test:
        #         tmp = []
        #         tmp.append(pred[0])
        #         myWriter.writerow(tmp)
        #
        # print ('done!')






def LSTM_pred_wd(X, Y, X_test=None, iters=1000, learning_rate=1e-1, dropout_prob=0.5):
    print ("lr:", learning_rate)
    print ("dropout:", dropout_prob)
    print ("iterations:", iters)
    num_count = Y.shape[0]
    input_shape = X.shape[1]
    print ('num_count:', num_count)
    print ('input_size:', input_shape)
    rnn = LSTM4wd(num_input=input_shape, num_hidden=34, num_output=1)

    Loss = []
    for iter in xrange(iters+1):
        loss = rnn.fit(X, Y, learning_rate, dropout_prob)
        Loss.append(loss)
        if iter % 2000 == 0:
            print ("iteration: %s, loss: %s" % (iter, loss))

    rnn.save_model_params('rnn_params_wd')

    if X_test is not None:
        print ('starting predicting......')
        Y_test = rnn.predict(X_test)
        print ('predicting done!')

        print ('done!')
        return Y_test

        # print ('writing result to csv')
        # with open('output/pred_eva_data.csv', 'wb') as MyFile:
        #     myWriter = csv.writer(MyFile)
        #     myWriter.writerow(["Evaluation"])
        #     for pred in Y_test:
        #         tmp = []
        #         tmp.append(pred[0])
        #         myWriter.writerow(tmp)
        #
        # print ('done!')


    

def LSTM_pred_dep(X, Y, X_test=None, iters=1000, learning_rate=1e-1, dropout_prob=0.5):
    print ("lr:", learning_rate)
    print ("dropout:", dropout_prob)
    print ("iterations:", iters)
    num_count = Y.shape[0]
    input_shape = X.shape[1]
    print ('num_count:', num_count)
    print ('input_size:', input_shape)
    rnn = LSTM4dep(num_input=input_shape, num_hidden=40, num_output=1)

    Loss = []
    for iter in xrange(iters+1):
        loss = rnn.fit(X, Y, learning_rate, dropout_prob)
        Loss.append(loss)
        if iter % 2000 == 0:
            print ("iteration: %s, loss: %s" % (iter, loss))

    print ('starting predicting......')
    Y_test = rnn.predict(X_test)
    print ('predicting done!')



    for i in range(60):
        if Y_test[i] <= 0:
            Y_test[i] = 0


    # print ('writing result to csv')
    # with open('output/pred_dep_data.csv', 'wb') as MyFile:
    #     myWriter = csv.writer(MyFile)
    #     myWriter.writerow(["Depth"])
    #     for pred in Y_test:
    #         tmp = []
    #         tmp.append(pred[0])
    #         myWriter.writerow(tmp)

    print ('done!')
    return Y_test



def doubleLSTM_pred_dep(X, Y, X_test=None, iters=1000, learning_rate=1e-1, dropout_prob=0.5):
    print("test successed!")
    print ("lr:", learning_rate)
    print ("dropout:", dropout_prob)
    print ("iterations:", iters)
    num_count = Y.shape[0]
    input_shape = X.shape[1]
    print ('num_count:', num_count)
    print ('input_size:', input_shape)
    rnn = doubleLSTM4dep(num_input=input_shape, num_hidden=40, num_output=1)

    Loss = []
    for iter in xrange(iters+1):
        loss = rnn.fit(X, Y, learning_rate, dropout_prob)
        Loss.append(loss)
        if iter % 2000 == 0:
            print ("iteration: %s, loss: %s" % (iter, loss))

    print ('starting predicting......')
    Y_test = rnn.predict(X_test)
    print ('predicting done!')



    for i in range(60):
        if Y_test[i] <= 0:
            Y_test[i] = 0


    # print ('writing result to csv')
    # with open('output/pred_dep_data.csv', 'wb') as MyFile:
    #     myWriter = csv.writer(MyFile)
    #     myWriter.writerow(["Depth"])
    #     for pred in Y_test:
    #         tmp = []
    #         tmp.append(pred[0])
    #         myWriter.writerow(tmp)

    print ('done!')
    return Y_test


def LSTM_eva(X, Y, X_test=None, iters=10000, learning_rate=1e-1, dropout_prob=0.5, index=0):
    print "lr:", learning_rate
    print "dropout:", dropout_prob
    print "iterations:", iters
    input_shape = X.shape[2]
    output_shape = Y.shape[2]
    rnn = LSTM4alleva(num_input=input_shape, num_hidden=40, num_output=output_shape)

    for iter in xrange(iters+1):
        total_loss = 0.0
        for i in range(len(X)):
            loss = rnn.fit(X[i], Y[i], learning_rate, dropout_prob)
            total_loss += loss
        if iter % 500 == 0:
            print "iteration: %s, loss: %s" % (iter, total_loss)

    rnn.save_model_params('../output/ave/eva_param'+str(index))


    print 'starting predicting......'
    PRED = []
    for i in range(len(X_test)):
        Y_test = rnn.predict(X_test[i])
        PRED.append(Y_test)
    print 'predicting done!'
    print 'shape', np.array(PRED).shape
    print 'writing results in file......'
    return PRED
    # with open('testfile.csv', 'wb') as MyFile:
    #     myWriter = csv.writer(MyFile)
    #     for predict in PRED:
    #         myWriter.writerow(predict)
    #
    # print 'done!'
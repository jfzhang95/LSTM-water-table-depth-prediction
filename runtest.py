#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date:   
"""




import numpy as np
import theano
import theano.tensor as T
from layers import *
from optim import adam
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from run_model import *

ss_X_eva_train = StandardScaler()
ss_X_eva_test = StandardScaler()
ss_y_eva_train = StandardScaler()

ss_X_wd_train = StandardScaler()
ss_X_wd_test = StandardScaler()
ss_y_wd_train = StandardScaler()

ss_X_dep_train = StandardScaler()
ss_X_dep_test = StandardScaler()
ss_y_dep_train = StandardScaler()




#####################################
##      这部分导入数据来预测蒸发量     ##
#####################################


# loading data
print("Loading data......")

df1 = pd.read_csv('processed/data_yigan.csv')
df2 = pd.read_csv('processed/data_jiefangzha.csv')
df3 = pd.read_csv('processed/data_yongji.csv')
df4 = pd.read_csv('processed/data_yichang.csv')
df5 = pd.read_csv('processed/data_wulate.csv')

alldata = pd.concat([df1,df2,df3,df4,df5],axis=0)

print("done!")



X = alldata.drop('Id',axis=1).drop('Year', axis=1) \
            .drop('Depth', axis=1).drop('Water_Discharge', axis=1)\
            .drop('Evaporation', axis=1).as_matrix()
y = alldata['Evaporation'].as_matrix().reshape(-1,1)


X_train_eva = X.reshape(5,168,4)[:,0:144]
X_test_eva = X.reshape(5,168,4)[:,144:]
y_train_eva = y.reshape(5,168,1)[:,0:144]

print "X_train_eva's shape", X_train_eva.shape
print "y_train_eva's shape", y_train_eva.shape
print "X_test_eva's shape", X_test_eva.shape

X_data = ss_X_eva_train.fit_transform(X)


X_train_eva_std = X_data.reshape(5,168,4)[:,0:144]
X_test_eva_std = X_data.reshape(5,168,4)

y_train = y_train_eva.reshape(720,1)
y_train_eva_std = ss_y_eva_train.fit_transform(y_train).reshape(5,144,1)


print "X_train_eva_std's shape", X_train_eva_std.shape
print "y_train_eva_std's shape", y_train_eva_std.shape
print "X_test_eva_std's shape", X_test_eva_std.shape


print("Starting predict eva......")

y_pred_std = LSTM_eva(X_train_eva_std, y_train_eva_std, X_test_eva_std, iters=5, learning_rate=1e-4)
y_pred = np.random.randn(5,168,1)
for i in range(len(X_test_eva_std)):
    y_pred[i] = ss_y_eva_train.inverse_transform(y_pred_std[i])
print(y_pred.shape)
print(y_pred)
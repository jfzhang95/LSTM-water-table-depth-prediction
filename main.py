import pandas as pd
from sklearn.preprocessing import StandardScaler
from rnn_model import *
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


ss_X_eva_train = StandardScaler()
ss_X_eva_test = StandardScaler()
ss_y_eva_train = StandardScaler()

ss_X_wd_train = StandardScaler()
ss_X_wd_test = StandardScaler()
ss_y_wd_train = StandardScaler()

ss_X_dep_train = StandardScaler()
ss_X_dep_test = StandardScaler()
ss_y_dep_train = StandardScaler()



def rmse(y1, y2):
    return np.sqrt(mean_squared_error(y1, y2))



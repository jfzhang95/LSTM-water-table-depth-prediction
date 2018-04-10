import pandas as pd
from sklearn.preprocessing import StandardScaler
from models import *
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

ss_X_dep = StandardScaler()
ss_y_dep = StandardScaler()

def rmse(y1, y2):
    return np.sqrt(mean_squared_error(y1, y2))

# Noted that the demo data are processed manually, so they are not real data,
# but they still can reflect the correlation between the original data.
data = pd.read_csv('demo.csv')

Inputs = data.drop('Year', axis=1).drop('Depth', axis=1)
Outputs = data['Depth']

Inputs = Inputs.as_matrix()
Outputs = Outputs.as_matrix().reshape(-1, 1)

# First 12 years of data
X_train_dep = Inputs[0:144]
y_train_dep = Outputs[0:144]

# Last 2 years of data
X_test_dep = Inputs[144:]

print("X_train_dep shape", X_train_dep.shape)
print("y_train_dep shape", y_train_dep.shape)
print("X_test_dep shape", X_test_dep.shape)

X = np.concatenate([X_train_dep, X_test_dep], axis=0)

# Standardization
X = ss_X_dep.fit_transform(X)

# First 12 years of data
X_train_dep_std = X[0:144]
y_train_dep_std = ss_y_dep.fit_transform(y_train_dep)

# All 14 years of data
X_test_dep_std  = X

model_restore = LSTM_FC_Model(num_input=5, num_hidden=[40], num_output=1)
# Loading model
print('Start loading model......')
model_restore.load_model_params('checkpoints/LSTM_FC_CKPT')
print('Model restored.')
print('Start predicting......')
y_pred_dep_ = model_restore.predict(X_test_dep_std)
y_pred_dep_ = ss_y_dep.inverse_transform(y_pred_dep_[144:])
print('Done.')

print('the value of R-squared of Evaporation is ', r2_score(Outputs[144:], y_pred_dep_))
print('the value of Root mean squared error of Evaporation is ', rmse(Outputs[144:], y_pred_dep_))

f, ax1 = plt.subplots(1, 1, sharex=True, figsize=(6, 4))

ax1.plot(Outputs[144:], color="blue", linestyle="-", linewidth=1.5, label="Measurements")
ax1.plot(y_pred_dep_, color="green", linestyle="--", linewidth=1.5, label="Proposed model")

# ax1.set_title('Results', fontsize=16, fontweight='normal')


plt.legend(loc='upper right')
plt.xticks(fontsize=8,fontweight='normal')
plt.yticks(fontsize=8,fontweight='normal')
plt.xlabel('Time (Month)', fontsize=10)
plt.ylabel('Water table depth (m)', fontsize=10)
plt.xlim(0, 25)
plt.savefig('results.png', format='png')
plt.show()
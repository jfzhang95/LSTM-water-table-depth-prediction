import pandas as pd
import numpy as np

data = pd.read_csv("data_yigan.csv")


year = data['Year'].as_matrix().reshape(-1, 1)
month = data['Month'].as_matrix().reshape(-1, 1)
irrigation = data['Irrigation'].as_matrix().reshape(-1, 1)
rainfall = data['Rainfall'].as_matrix().reshape(-1, 1)
tem = data['Tem'].as_matrix().reshape(-1, 1)
evaporation = data['Evaporation'].as_matrix().reshape(-1, 1)
depth = data['Depth'].as_matrix().reshape(-1, 1)
print(np.shape(rainfall))

irr_mean = np.mean(irrigation) / 10
rf_mean = np.mean(rainfall) / 10
tem_mean = np.mean(tem) / 10
eva_mean = np.mean(evaporation) / 10
dep_mean = np.mean(depth) / 10

irrigation += np.random.randn(168, 1) * irr_mean
rainfall += np.random.randn(168, 1) * rf_mean
tem += np.random.randn(168, 1) * tem_mean
evaporation += np.random.randn(168, 1) * eva_mean
depth += np.random.randn(168, 1) * dep_mean

irrigation *= 4
rainfall *= 4
tem *= 4
evaporation *= 4
depth*= 4

DATA = np.concatenate([year, month, irrigation, rainfall, tem, evaporation, depth], axis=1)
to_data = pd.DataFrame(DATA, columns=['Year', 'Month', 'Irrigation', 'Rainfall', 'Tem', 'Evaporation', 'Depth'])
to_data.to_csv('demo.csv')

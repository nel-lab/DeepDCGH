#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 07:10:08 2021

@author: hoss
"""

from sklearn.linear_model import LinearRegression, Lasso
from sklearn import svm
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from joblib import dump, load


csv_folder = './CSV_Files/'
dataframe = pd.read_csv(csv_folder+'Main_withIQA.csv')

df_forML = dataframe.drop(columns=['Method_Names', 'Iterations', 'Methods'])

mos_names = ['Aram', 'Hoss', 'Nick', 'April']
df_forML['MOS'] = df_forML[mos_names].mean(axis=1)
df_forML = df_forML.drop(columns=mos_names)

y = np.squeeze(np.array(df_forML.loc[:, ['MOS']]).astype(np.float32)-0.5)/5
y -= y.min()
y /= y.max()
y += 0.13
y /= 1.3
X = df_forML.drop(columns=['MOS', 'PSNR', 'sewar_PSNR', 'sewar_PSNRB', 'sewar_RASE', 'MSSSIM', 'sewar_RMSE'])

num_feats = 5

fs = SelectKBest(score_func=f_regression, k=num_feats)
fs.fit(X, y)
X_fs = fs.transform(X)
#%
X_tr, X_te, y_tr, y_te = train_test_split(X_fs, y, test_size=0.3, random_state=42)
y_tr += 0.05*np.random.uniform(-1,1,y_tr.shape)

#%
for i in range(len(fs.scores_)):
    print(X.columns[i], fs.scores_[i])
#%
model = tf.keras.Sequential([
    layers.BatchNormalization(input_shape = (num_feats,)),
    layers.Dense(units=num_feats*2, activation=None, kernel_regularizer=regularizers.l2(1e-2)),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    # layers.Dense(units=num_feats*2, activation='relu'),
    # layers.Dropout(0.25),
    # layers.BatchNormalization(),
    layers.Dense(units=1, activation='sigmoid')])

#%
model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics = ['mse', 'mae'],
              experimental_run_tf_function=False)

#%
model.optimizer.lr = 0.1
model.fit(X_tr,
          y_tr,
          batch_size = 128,
          epochs = 5,
          validation_data = (X_te, y_te),
          verbose=2)

model.optimizer.lr = 0.01
model.fit(X_tr,
          y_tr,
          batch_size = 128,
          epochs = 10,
          validation_data = (X_te, y_te),
          verbose=2)

model.optimizer.lr = 0.001
model.fit(X_tr,
          y_tr,
          batch_size = 128,
          epochs = 10,
          validation_data = (X_te, y_te),
          verbose=2)

#%
model.optimizer.lr = 0.00001
model.fit(X_tr,
          y_tr,
          batch_size = 128,
          epochs = 300,
          validation_data = (X_te, y_te),
          verbose=2)

#%%
preds = []
# preds.append(np.squeeze(model.predict(X_fs)))

#%
regr = svm.SVR(kernel='linear')
regr.fit(X_tr, y_tr)
preds.append(regr.predict(X_fs))
dump(regr, './models/regr.joblib')
#%
n_neighbors = 5
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
preds.append(knn.fit(X_tr, y_tr).predict(X_fs))
dump(knn, './models/knn.joblib')
#%
dt = tree.DecisionTreeRegressor()
dt = dt.fit(X_tr, y_tr)
preds.append(dt.predict(X_fs))
dump(dt, './models/dt.joblib')
#%
las = LassoCV()
las = las.fit(X_tr, y_tr)
preds.append(las.predict(X_fs))
dump(las, './models/las.joblib')
#%
rig = RidgeCV()
rig = rig.fit(X_tr, y_tr)
preds.append(rig.predict(X_fs))
dump(rig, './models/rig.joblib')
#%
newX_tr = np.array(preds).T

#%
preds2 = np.mean(newX_tr, axis=-1)

print(np.corrcoef(preds2, y))
corr_mat = df_forML.corr()

plt.scatter(y, preds2)
plt.xlabel('Ground Truth MOS')
plt.ylabel('Predicted MOS')
plt.show()

#%%






















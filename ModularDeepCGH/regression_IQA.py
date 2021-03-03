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

csv_folder = './CSV_Files/'
dataframe = pd.read_csv(csv_folder+'Main_withIQA.csv')

df_forML = dataframe.drop(columns=['Method_Names', 'Iterations', 'Methods'])
df_forML['MOS'] = (np.array(df_forML['Aram']) + np.array(df_forML['Hoss']))/2.
df_forML = df_forML.drop(columns=['Hoss', 'Aram'])

#%%
feat_columns = ['SSIM', 'NIQE_GT', 'Accuracy', 'BRISQUE_DIFF', 'NIQUE_DIFF']
X_tr = df_forML.loc[:499, feat_columns]
y_tr = df_forML.loc[:499, ['MOS']].astype(np.float32)/5 - (1/5)
X_te = df_forML.loc[500:, feat_columns]
y_te = df_forML.loc[500:, ['MOS']].astype(np.float32)/5 - (1/5)
# X = df_forML.loc[:, feat_columns]
# y = df_forML.loc[:, ['MOS']].astype(np.float32)/5 - (1/5)
y = np.squeeze(np.array(df_forML.loc[:, ['MOS']]).astype(np.float32)/5 - (1/5))
X = df_forML.drop(columns=['MOS', 'PSNR'])

#%%
# model = Lasso()
model = LinearRegression()
# model = svm.SVR()

#%%
cv_results = cross_validate(model, X, y, cv=5)

#%%
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

feat_columns = ['SSIM', 'MAE', 'Accuracy', 'NIQUE_DIFF', 'NIQE']
X_tr = np.array(df_forML.loc[:499, feat_columns])#.drop(columns=['MOS', 'PSNR'])
y_tr = np.squeeze(np.array(df_forML.loc[:499, ['MOS']]).astype(np.float32)/5 - (1/5))
X_te = np.array(df_forML.loc[500:, feat_columns])
y_te = np.squeeze(np.array(df_forML.loc[500:, ['MOS']]).astype(np.float32)/5 - (1/5))

#%%
model = tf.keras.Sequential([
    layers.Dense(input_shape = (5,), units=10, activation=None),
    layers.Dropout(0.25),
    layers.BatchNormalization(),
    layers.Dense(units=10, activation='relu'),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer = 'adam',
              loss = 'mae',
              metrics = ['mse'],
              experimental_run_tf_function=False)

model.optimizer.lr = 0.001

#%%
model.fit(X_tr,
          y_tr,
          batch_size = 64,
          epochs = 10000,
          validation_data = (X_te, y_te),
          verbose=2)

#%%
preds = np.squeeze(model.predict(X_te))

#%%
print(np.corrcoef(preds, y_te))
corr_mat = df_forML.corr()

#%%
import matplotlib.pyplot as plt
plt.hist(y_te, 100)
plt.show()

#%%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

fs = SelectKBest(score_func=f_regression, k='all')

fs.fit(X_tr, y_tr)

plt.figure(figsize=(20, 10))
plt.bar(list(df_forML.loc[:499, :].drop(columns=['MOS', 'PSNR']).columns), fs.scores_)
plt.show()
























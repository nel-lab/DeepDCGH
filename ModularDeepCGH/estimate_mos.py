#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:16:54 2021

@author: hoss
"""
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from joblib import dump, load
from glob import glob
import numpy as np

#%%
image_path = ''
files = glob(image_path+'*.jpg')


#%%
regr = load('regr.joblib')
knn = load('knn.joblib')
dt = load('dt.joblib')
regr = load('regr.joblib')
regr = load('regr.joblib')

#%%
init = 10000
prof = 0.02
for i in range(12*24):
    a = np.random.randint(-2,5)
    init *= (1+(prof*a))
print(init)

#%%
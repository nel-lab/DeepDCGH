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
from sewar.full_ref import sam, vifp
from utils import tf_ssim, rmse_sw, accuracy

#%%

class MOS_Estimator(object):
    def __init__(self, path_to_models):
        files = glob(path_to_models+'/*.joblib')
        self.models = []
        self.metrics = [sam, vifp, tf_ssim, rmse_sw, accuracy]
        self.metric_names = ['SAM','VIF','SSIM','RMSE','ACC']
        self.scores = {}
        self.moses = {}
        for file in files:
            self.models.append(load(file))
        
    def get_mos(self, img, cgh):
        scores = []
        
        img = np.squeeze(img).astype(np.float32).copy()
        cgh = np.squeeze(cgh).astype(np.float32).copy()
        
        for metric, name in zip(self.metrics, self.metric_names):
            scores.append(metric(img, cgh))
            self.scores[name] = scores[-1]
        
        scores = np.array(scores)[np.newaxis, ...]
        
        mos_ls = []
        for model in self.models:
            mos_ls.append(model.predict(scores))
            
        self.moses = mos_ls
        self.mos = np.mean(np.array(mos_ls))
        return self.mos

#%%

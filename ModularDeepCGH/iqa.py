#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
accuracy
psnr
ssim
perceptual
fsim
KonCept512

"""
from tqdm import tqdm
from skvideo.measure import ssim
import tensorflow as tf
from skvideo.measure import mse
from skvideo.measure import mae
from skvideo.measure import psnr
import os
from skvideo.measure import viideo_score
from skvideo.measure import niqe
import pandas as pd
from utils import load_scores_fromCSV, tf_msssim, brisque, accuracy_batch, normalize_minmax
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

#%%
csv_folder = './CSV_Files/'
image_path = './image_data.h5'
dataframe = load_scores_fromCSV(csv_folder)

#%%
with h5.File(image_path, 'r') as f:
    targets = f['OG'][:]
    cghs = f['Amplitudes'][:]

fr_names = ['SSIM', 'MSSSIM', 'MSE', 'MAE', 'PSNR']
nr_names = ['BRISQUE','NIQE']
fr_methods = [ssim, tf_msssim, mse, mae, psnr]
nr_methods = [brisque, niqe]

#%%                                    FR              NR
if os.path.isfile(csv_folder+'Main_withIQA.csv'):
    dataframe = pd.read_csv(csv_folder+'Main_withIQA.csv')
else:
    scores = np.zeros((len(cghs), len(fr_names) + len(nr_names)), dtype=np.float32)
    score_sanity = np.zeros((len(cghs), len(nr_names)), dtype=np.float32)
    
    for i, target, cgh in tqdm(zip(range(len(cghs)), targets, cghs), total=len(cghs)):
        for j1, fr_method in zip(range(len(fr_names)), fr_methods):
            scores[i, j1] = fr_method(target, cgh)
        
        for j2, nr_method in zip(range(len(nr_names)), nr_methods):
            scores[i, j1+j2+1] = nr_method(cgh)
            score_sanity[i, j2] = nr_method(target)
            
    for name, array in zip(fr_names+nr_names, scores.T):
        dataframe[name] = array
        
    for name, array in zip([name+'_GT' for name in nr_names], score_sanity.T):
        dataframe[name] = array

    dataframe['Accuracy'] = accuracy_batch(normalize_minmax(targets), normalize_minmax(cghs)).numpy()

    dataframe['BRISQUE_DIFF'] = np.abs(np.array(dataframe['BRISQUE_GT']) - np.array(dataframe['BRISQUE']))
    dataframe['NIQUE_DIFF'] = np.abs(np.array(dataframe['NIQE_GT']) - np.array(dataframe['NIQE']))

#%% VIF


#%% perceptual


#%% Koncept


#%% MetaIQA


#%%
df_forML = dataframe.drop(columns=['Method_Names', 'Iterations', 'Methods'])
df_forML['MOS'] = (np.array(df_forML['Aram']) + np.array(df_forML['Hoss']))/2.
df_forML = df_forML.drop(columns=['Hoss', 'Aram'])

#%%
corr_mat = df_forML.corr()

#%%































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
from tensorflow.keras.models import Model
from skvideo.measure import mse
from skvideo.measure import mae
from skvideo.measure import psnr
import os
from skvideo.measure import viideo_score
from skvideo.measure import niqe
import pandas as pd
from utils import load_scores_fromCSV, tf_msssim, brisque, accuracy_batch, normalize_minmax, rmse_sw
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

#%%
csv_folder = './CSV_Files/'
image_path = './image_data.h5'
old_df = load_scores_fromCSV(csv_folder)

#%%
with h5.File(image_path, 'r') as f:
    targets = f['OG'][:]
    cghs = f['Amplitudes'][:]

#%%
if os.path.isfile(csv_folder+'Main_withIQA.csv'):
    dataframe = pd.read_csv(csv_folder+'Main_withIQA.csv')
else:
    fr_names = ['SSIM', 'MSSSIM', 'MSE', 'MAE', 'PSNR']
    nr_names = ['BRISQUE','NIQE']
    fr_methods = [ssim, tf_msssim, mse, mae, psnr]
    nr_methods = [brisque, niqe]
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


#%% perceptual
image_shape = cghs.shape[1:] + (3,)

vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
vgg19.trainable = False
for l in vgg19.layers:
    l.trainable = False
loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
loss_model.trainable = False

cghs_features = loss_model.predict(np.broadcast_to(cghs[..., np.newaxis].astype(np.float32)/255., shape=(len(cghs),)+image_shape), batch_size=16)
target_features = loss_model.predict(np.broadcast_to(targets[..., np.newaxis].astype(np.float32)/255., shape=(len(cghs),)+image_shape), batch_size=16)
final = tf.reduce_mean(tf.square(cghs_features - target_features), axis=[1,2,3]).numpy()
dataframe['Perceptual'] = final


#%% Koncept Color trained
from koncept.models import Koncept512
k = Koncept512()
kon_cgh = k.assess(np.broadcast_to(targets[..., np.newaxis], shape=targets.shape+(3,)))
kon_gt = k.assess(np.broadcast_to(cghs[..., np.newaxis], shape=targets.shape+(3,)))
dataframe['KonCept'] = kon_cgh
dataframe['KonCept_GT'] = kon_gt
dataframe['KonCept_DIFF'] = np.abs(kon_gt - kon_cgh)
dataframe['KonCept_RAT'] = np.abs(kon_cgh / kon_gt)


#%% Koncept Gray trained
kon_cgh = np.load('GKoncept.npz')['GKoncept']
kon_gt = np.load('GKoncept.npz')['GKoncept_GT']
dataframe['GKonCept'] = kon_cgh
dataframe['GKonCept_GT'] = kon_gt
dataframe['GKonCept_DIFF'] = np.abs(kon_gt - kon_cgh)

#%% VIF, SEWAR
from sewar.full_ref import ergas, psnr, psnrb, rase, rmse, sam, scc, uqi, vifp

names = ['sewar_ERGAS', 'sewar_PSNR', 'sewar_PSNRB', 'sewar_RASE', 'sewar_RMSE', 'sewar_RMSE_SW', 'sewar_SAM', 'sewar_SCC', 'sewar_UQI', 'sewar_VIFP']
methods = [ergas, psnr, psnrb, rase, rmse, rmse_sw, sam, scc, uqi, vifp]

scores = np.zeros((len(cghs), len(names)), dtype=np.float32)

for i, target, cgh in tqdm(zip(range(len(cghs)), targets, cghs), total=len(cghs)):
    for j, method in zip(range(len(names)), methods):
        scores[i, j] = method(target, cgh)

for name, array in zip(names, scores.T):
    dataframe[name] = array


#%% MetaIQA



#%%
df_forML = dataframe.drop(columns=['Method_Names', 'Iterations', 'Methods'])
df_forML['MOS'] = (np.array(df_forML['Aram']) + np.array(df_forML['Hoss']))/2.
df_forML = df_forML.drop(columns=['Hoss', 'Aram'])
df_forML.to_csv(csv_folder+'Main_withIQA_MLReady.csv', index=False)

#%%
corr_mat = df_forML.corr()


#%%
dataframe.to_csv(csv_folder+'Main_withIQA.csv', index=False)

#%%
dataframe = pd.from_csv()

#%%














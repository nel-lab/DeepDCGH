#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:18:10 2020

@author: hoss
"""
dev_num = 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_num)
import tensorflow as tf
from deepcgh import DeepCGH_Datasets, DeepCGH
import h5py as h5
import numpy as np
from tqdm import tqdm
from utils import lens
from glob import glob
retrain = True
coordinates = False

N = 1000
size = (512, 512)
del_existing = True
image_path = '/storage1/datasets/natural_images/COCO/train2017'
filename = '/nvme/datasets/natural_images/COCO/COCO2017_Size{}_N{}.h5'.format(size[0], N)

data = {
        'path' : 'DeepCGH_Datasets/Disks',
        'shape' : (512, 512, 1),
        # 'shape' : (slm.height, slm.width, 3),
        'object_type' : 'Mix',
        'object_size' : 10,
        'object_count' : [270, 480],
        'intensity' : [0.2, 1],
        'normalize' : True,
        'centralized' : False,
        'N' : 40000,
        'train_ratio' : 39000/40000,
        'compression' : 'GZIP',
        'name' : 'target',
        }

model = {
        'path' : 'DeepCGH_Models/Disks',
        'int_factor':16,
        'n_kernels':[ 64, 128, 256],
        'plane_distance':0.005,
        'wavelength':1e-6,
        'pixel_size':0.000015,
        'input_name':'target',
        'output_name':'phi_slm',
        'lr' : 1e-4,
        'batch_size' : 4,
        'epochs' : 10,
        'token' : '64',
        'shuffle' : 8,
        'max_steps' : 40000
        }

# Get data
dset = DeepCGH_Datasets(data)

dset.getDataset()

# Estimator
dcgh = DeepCGH(data, model)

dcgh.train(dset)

#%%
names = ["DeepCGH_{}_P".format(data['object_type'])]
# names.append("DeepCGH{}_A".format(data['object_type']))
dtype = np.uint8
mul = -1+2**8
with h5.File(filename, 'a') as f:
    dsets = {}

    keys = list(f.keys())
    if del_existing and names[0] in keys:
        print('Deleting old dataset...')
        del f[names[0]]

    ogs = f['OG']
    for name in names:
        dsets[name] = f.create_dataset(name, shape=(ogs.shape[0],) + size, dtype=dtype)
    for ind in tqdm(range(N)):
        img = ogs[ind].astype(np.float32)
        img /= img.max()

        ph = np.squeeze(dcgh.get_hologram(img[np.newaxis, ..., np.newaxis]))
        ph = np.mod(ph, 2*np.pi)
        ph -= ph.min()
        ph /= ph.max()
        ph *= mul
        dsets["DeepCGH_{}_P".format(data['object_type'])][ind] = np.round(ph).astype(dtype)
        # dsets["DeepCGH{}_A".format(data['object_type'])][ind] = lens(ph)
































































































































































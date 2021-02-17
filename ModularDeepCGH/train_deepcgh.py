#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:18:10 2020

@author: hoss
"""
dev_num = 2
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

N = 50000
size = (512, 512)
del_existing = False
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

if retrain:
    dcgh.train(dset)

#%%
names = []
names.append("DeepCGH{}_A".format(data['object_type']))
names.append("DeepCGH{}_P".format(data['object_type']))

with h5.File(filename, 'a') as f:
    dsets = {}

    keys = list(f.keys())

    # check if dataset already exists and inform
    for k in keys:
        if k in names:
            print('{} already exists.'.format(k))
            if del_existing:
                print('Deleting old dataset...')
                del f[k]
            else:
                print('Keeping old dataset.')
                names.remove(k)

    assert len(names) != 0, "Data already exists in the specified file:"

    for name in names:
        dsets[name] = f.create_dataset(name, shape=(N,) + size, dtype=np.float32)

    for ind in tqdm(range(N)):
        img = f['OG'][ind].astype(np.float32)
        img -= img.min()
        img /= img.max()

        ph = np.squeeze(dcgh.get_hologram(img[np.newaxis, ..., np.newaxis]))
        dsets["DeepCGH{}_A".format(data['object_type'])][ind] = ph
        dsets["DeepCGH{}_A".format(data['object_type'])][ind] = lens(ph)
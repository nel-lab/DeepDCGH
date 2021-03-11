#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:54:04 2021

@author: hoss
"""

dev_num = 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_num)
import numpy as np
import h5py as h5
from PIL import Image
import pandas as pd
from sgd_holography import novocgh
from utils import gs
from tqdm import tqdm
import matplotlib.pyplot as plt
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
def get_uint8(data):
    data = data.astype(np.float32)
    data -= data.min()
    data /= data.max()
    data *= 255
    return np.round(data).astype('uint8')

def normalize(data):
    data = data.astype(np.float32)
    data -= data.min()
    return data / data.max()

dataframe_path = "/nvme/datasets/natural_images/koniq/koniq10k_scores_and_distributions.csv"
image_folder = '/nvme/datasets/natural_images/koniq/1024x768/'
hdf5_file = '/nvme/datasets/natural_images/koniq/KonIQ_{}_{}_{}+.h5'

df = pd.read_csv(dataframe_path)

filenames = df['image_name'][df['MOS']>3.7][1000:]
size = (768//2, 1024//2)


with h5.File(hdf5_file.format(size[0], size[1], len(filenames)), 'w') as f:
    images = f.create_dataset('OG', shape = (len(filenames),)+size, dtype = np.uint8)
    amps = f.create_dataset('Amplitudes_{}'.format(data['object_type']), shape = (len(filenames),)+size, dtype = np.uint8)
    for ind, name in tqdm(zip(range(len(filenames)), filenames), total=len(filenames)):
        img = normalize(np.mean(np.array(Image.open(image_folder+name).resize(size[::-1])), axis=-1))
        images[ind] = get_uint8(img)
        phase = get_uint8(normalize(lens(np.squeeze(dcgh.get_hologram(img[np.newaxis, ..., np.newaxis])))))
        amps[ind] = phase
        
        
#%%

















































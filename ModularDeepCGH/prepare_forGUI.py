#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 06:54:37 2021

@author: hoss
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import h5py as h5
import numpy as np
file = '/home/hoss/Documents/datasets/COCO2017_Size512_N1000.h5'
cghs = []
names = []

with h5.File(file, 'r') as f:
    images = f['OG'][:].astype(np.float32)
    for k in f.keys():
        names.append(k)
        cghs.append(f[k][:])
names.remove('OG')
#%%
def fft(phase): 
    slm_cf = tf.math.exp(tf.complex(0., phase)) 
    img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf)) 
    img = tf.math.abs(img_cf) 
    return img 

#%%
dsets = []
file2 = '/home/hoss/Documents/datasets/ForGUI.h5'
images -= np.min(images, axis = (1,2), keepdims=True)
images /= np.max(images, axis = (1,2), keepdims=True)
images *= -1+2**8

#%%
with h5.File(file2, 'w') as f:
    ogs = f.create_dataset(name='OG', data=np.round(images).astype(np.uint8))
    for cgh, name in zip(cghs, names):
        dsets.append(f.create_dataset(name, shape=ogs.shape, dtype=np.uint8))
        for ind, phi in enumerate(cgh):
            phi = (phi.astype(np.float32) / phi.max())*2*np.pi
            img_ = fft(phi).numpy()
            img_ -= img_.min()
            img_ /= img_.max()
            img_ *= -1+2**8
            dsets[-1][ind] = np.round(img_).astype(np.uint8)

#%%
plt.imshow(images[0])
plt.show()














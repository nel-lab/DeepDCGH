#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:48:22 2021

@author: hoss
"""
import tensorflow as tf

@tf.function
def __normalize_minmax(img):
    img = tf.cast(img, tf.float32)
    img -= tf.reduce_min(img, axis=[0, 1], keepdims=True)
    img /= tf.reduce_max(img, axis=[0, 1], keepdims=True)
    return img

@tf.function
def __gs(img):
    rand_phi = tf.random.uniform(img.shape)
    img = __normalize_minmax(img)
    img_cf = tf.complex(img, 0.) * tf.math.exp(tf.complex(0., rand_phi))
    slm_cf = tf.signal.ifft2d(tf.signal.ifftshift(img_cf))
    slm_phi = tf.math.angle(slm_cf)
    return slm_phi


def __accuracy(y_true, y_pred):
    denom = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis = [0, 1])*tf.reduce_sum(tf.pow(y_true, 2), axis = [0, 1]))
    return 1 - (tf.reduce_sum(y_pred * y_true, axis = [0, 1])+1)/(denom+1)


def novocgh(img, Ks, lr = 0.1):
    slms = []
    amps = []
    phi = __gs(img)
    phi_slm = tf.Variable(phi)
    opt = tf.keras.optimizers.Adam(learning_rate = lr)
    img = tf.convert_to_tensor(img)

    def loss_(phi_slm):
        slm_cf = tf.math.exp(tf.complex(0., phi_slm))
        img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
        return tf.math.abs(img_cf)

    def loss():
        slm_cf = tf.math.exp(tf.complex(0., phi_slm))
        img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
        amp = tf.math.abs(img_cf)
        return __accuracy(tf.square(img), tf.square(amp))

    for i in range(Ks[-1]+1):
        opt.minimize(loss, var_list=[phi_slm])
        if i in Ks:
            amps.append(loss_(phi_slm).numpy())
            slms.append(phi_slm.numpy())
    return slms, amps


#%


#%%
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import h5py as h5
# import numpy as np
# file = '/home/hoss/Documents/COCO2017_Size512_N50.h5'
# cghs = []
# names = []
# with h5.File(file, 'r') as f:
#     images = f['OG'][:]
#     for k in f.keys():
#         if k != 'OG':
#             names.append(k)
#             cghs.append(f[k][:])
# img = images[10].astype(np.float32)
# img /= img.max()
#
# #%%
# plt.imshow(img)
# plt.show()
#
# #%%
# def fft(phase):
#     slm_cf = tf.math.exp(tf.complex(0., phase))
#     img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
#     img = tf.math.abs(img_cf)
#     return img
#
# #%%
# final_slms = []
# slms, amps = novocgh(img, [1,15,40,100], lr=0.1)
#
# #%%
# for it, phiii in enumerate(slms):
#     # phi = (phiii.astype(np.float32) / phiii.max())*2*np.pi
#     img_ = fft(phiii).numpy()
#     plt.imshow(img_)
#     plt.title(it)
#     plt.show()
# plt.imshow(img)
# plt.show()
    
#%%













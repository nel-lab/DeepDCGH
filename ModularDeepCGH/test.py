#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:31:32 2021

@author: hoss
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import h5py as h5

#%%
filename = '/home/hoss/Documents/datasets/ForGUI.h5'

with h5.File(filename, 'r') as f:
    for ind, k in enumerate(f.keys()):
        plt.figure(figsize=(5,5))
        plt.imshow(f[k][51])
        plt.axis('off')
        plt.title(k)
        plt.savefig(k+'.png')
        plt.show()


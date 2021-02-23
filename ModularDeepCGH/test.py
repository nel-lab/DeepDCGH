#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:31:32 2021

@author: hoss
"""

import numpy as np
import random

imgs = list(range(10))
order = np.zeros((10*4, 2))
count = 0

for i in range(10):
    for j in range(4):
        order[count, 0] = i
        order[count, 1] = j
        count+=1
order = list(order)
random.shuffle(order_)

#%%
img_index, method_index = order_[10]

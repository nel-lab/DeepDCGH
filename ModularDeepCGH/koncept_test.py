#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:05:02 2021

@author: hoss
"""

from kuti import applications as apps
from kuti import generic as gen
from kuti import image_utils as iu
import pandas as pd, numpy as np, os, urllib

#%%
# download and read the meta-data for the KonIQ-10k IQA database
koniq_meta_url = "https://github.com/subpic/koniq/raw/master/metadata/koniq10k_distributions_sets.csv"
urllib.request.urlretrieve(koniq_meta_url, 'koniq10k_distributions_sets.csv')
df = pd.read_csv('koniq10k_distributions_sets.csv')

# download some images from the test set of the database via direct link
url_list = 'http://media.mmsp-kn.de/koniq10k/1024x768/' + df[df.set=='test'].image_name[::50]
gen.make_dirs('tmp/')
for url in url_list:
    file_name = url.split('/')[-1]
    urllib.request.urlretrieve(url, 'tmp/'+file_name)

#%%
import koncept
from koncept.models import Koncept512
k = Koncept512()

#%%
# read images and assess their quality
images = [np.broadcast_to(np.mean(iu.read_image(p).astype(np.float32), axis=-1, keepdims=True), shape=(768, 1024, 3))/255 for p in 'tmp/' + df[df.set=='test'].image_name[::50]]

#%%
images = np.zeros((12, 384, 512, 3))
MOS_pred = k.assess(images)

#%% compare with the ground-truth quality mean opinion scores (MOS)
MOS_ground = df[df.set=='test'].MOS[::50]
apps.rating_metrics(MOS_ground, MOS_pred);

#%%
import matplotlib.pyplot as plt
plt.imshow(images[0])
plt.show()

#%%

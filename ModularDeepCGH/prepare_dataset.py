dev_num = 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_num)
from utils import create_datasets, gs
from sgd_holography import novocgh
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

N = 1000
size = (512, 512)
image_path = '/storage1/datasets/natural_images/COCO/train2017'
filename = '/nvme/datasets/natural_images/COCO/COCO2017_Size{}_N{}.h5'.format(size[0], N)

Ks_GS = [1, 100]#list(range(0, 201, 5))
Ks_NOVO = [20, 200]#list(range(0, 502, 100))

#% GS algorithm
create_datasets(gs,
                image_path,
                filename,
                'GS',
                ks = Ks_GS,
                img_format = 'jpg',
                shape = size,
                del_existing = True,
                phase_only = True,
                N = N,
                dtype = np.uint8)

#% NOVO-CGH algorithm
create_datasets(novocgh,
                image_path,
                filename,
                "NOVO",
                ks = Ks_NOVO,
                img_format = 'jpg',
                shape = size,
                del_existing = True,
                phase_only = True,
                delete=False,
                N = N,
                dtype = np.uint8)

#%%
with h5.File(filename, 'a') as f:
    print(f['OG'].shape)
    # keys = list(f.keys())
    # if len(keys) != 0:
    #     print('Keys are {}'.format(keys))
    # imgs = f['OG'][:]
    # phis = f['NOVO_P_100'][:]
    # amps = f['NOVO_A_100'][:]

#%%
for i in range(2):
    plt.imshow(imgs[i])
    plt.show()
    plt.imshow(amps[i])
    plt.show()















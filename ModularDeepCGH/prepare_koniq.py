import numpy as np
import h5py as h5
from PIL import Image
import pandas as pd
from sgd_holography import novocgh
from utils import gs
from tqdm import tqdm
import matplotlib.pyplot as plt

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
hdf5_file = '/nvme/datasets/natural_images/koniq/KonIQ_{}_{}_{}.h5'

df = pd.read_csv(dataframe_path)

filenames = df['image_name'][df['MOS']>3.7]
size = (768//2, 1024//2)
methods = ['GS', 'NOVO', 'original']
gs_iters = np.array(list(range(1, 50, 10)))
num_gs = len(gs_iters)
novo_iters = np.array(list(range(0, 200, 50)))
num_novo = len(novo_iters)

methods = np.random.randint(0, 5, (len(filenames),))

with h5.File(hdf5_file.format(size[0], size[1], len(filenames)), 'w') as f:
    images = f.create_dataset('OG', shape = (len(filenames),)+size, dtype = np.uint8)
    amps = f.create_dataset('Amplitudes', shape = (len(filenames),)+size, dtype = np.uint8)
    methods_dset = f.create_dataset('Methods_GS_NOVO_OG', data = methods)
    iters_dset = f.create_dataset('Iters', shape=(len(filenames),), dtype=np.uint16)
    for ind, name, method in tqdm(zip(range(len(filenames)), filenames, methods), total=len(filenames)):
        img = normalize(np.mean(np.array(Image.open(image_folder+name).resize(size[::-1])), axis=-1))
        images[ind] = get_uint8(img)
        
        if method in [0, 1]: # it's GS
            iters_dset[ind] = gs_iters[np.random.randint(0, num_gs)]
            _, amps_gs = gs(img, [int(iters_dset[ind])])
            amps[ind] = get_uint8(amps_gs[0].numpy())
        elif method in [2,3]: # it's NOVO
            iters_dset[ind] = novo_iters[np.random.randint(0, num_novo)]
            _, amps_novo = novocgh(img, [int(iters_dset[ind])], lr = 0.5)
            amps[ind] = get_uint8(amps_novo[0])
        else:
            amps[ind] = get_uint8(img)
            iters_dset[ind] = 0

#%%
with h5.File(hdf5_file.format(len(filenames)), 'r') as f:
    for k in f.keys():
        print(k)

#%%
with h5.File(hdf5_file.format(len(filenames)), 'r') as f:
    Iters = f['Iters'][:]
    Methods = f['Methods_GS_NOVO_OG'][:]
    Amplitudes = f['Amplitudes'][:]
    OG = f['OG'][:]

#%%
ind = 12

plt.imshow(OG[ind])
plt.show()
plt.imshow(Amplitudes[ind])
plt.title('Method {} and iters {}'.format(Methods[ind], Iters[ind]))
plt.show()

#%%
plt.imshow(img)
plt.show()
























































import numpy as np
import h5py as h5
from PIL import Image
from tqdm import tqdm
from glob import glob
import os
import tensorflow as tf
import imquality.brisque as brisque_
import pandas as pd
from sewar.full_ref import rmse_sw as rmse_sw_


def rmse_sw(gt, p):
    return rmse_sw_(gt, p)[0]


def smartResize(img, target_size):
    cur_shape = img.shape[:-1]
    min_size = min(cur_shape)
    start = (np.array(cur_shape) - np.array((min_size,)*2)) // 2
    crop_img = Image.fromarray(img[start[0] : start[0] + min_size, start[1] : start[1] + min_size])
    img = np.array(crop_img.resize(target_size)).astype(np.float32)
    img -= img.min()
    img /= img.max()
    return img

def smartDType(img_, dtype):
    img = img_.copy().astype(np.float64)
    try:
        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max
        if min_val != 0:
            img *= 2
            img -= 1
            img *= max_val//3.
        else:
            img *= max_val
    except:
        min_val = np.finfo(dtype).min
        max_val = np.finfo(dtype).max
        if min_val != 0:
            img *= 2
            img -= 1
            img *= max_val//3.
        else:
            img *= max_val
    return img.astype(dtype)


def loadImageasGray(file, shape, dtype):
    img = np.array(Image.open(file))
    if len(img.shape) > 2:
        img = np.mean(img, axis=-1)

    if img.shape != shape:
        img = smartResize(img, shape)
    
    return smartDType(img, dtype)
    

def create_datasets(method,
                    image_path,
                    filename,
                    dset_name,
                    ks = [0, 10, 20],
                    img_format = 'jpg',
                    shape = (512,512),
                    del_existing = False,
                    phase_only = True,
                    delete = True,
                    N = -1,
                    dtype = np.uint16):
    
    if dtype == np.uint8:
        mul = 255
    elif dtype == np.uint16:
        mul = -1+2**16
    else:
        print("dtype not supported. Using uint16")
        dtype = np.uint16
        mul = -1+2**16
    
    if delete and os.path.isfile(filename):
        os.remove(filename)

    Ks = ks.copy()
    files = sorted(glob(os.path.join(image_path, '*.' + img_format)))
    print("{} files found.".format(len(files)))
    if N == -1: # it means all files should be processed
        N = len(files)

    names = []
    if 'DeepCGH' not in dset_name:
        for k in Ks:
            names.append("{}_P_{}".format(dset_name, str(k)))
            if not phase_only:
                names.append("{}_A_{}".format(dset_name, str(k)))
    else:
        names.append("{}_P".format(dset_name))
        if not phase_only:
            names.append("{}_A".format(dset_name))

    with h5.File(filename, 'a') as f:

        dsets = {}

        keys = list(f.keys())

        # check whether OG needs saving too
        if len(keys) != 0:
            og = False
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
                        try:
                            Ks.remove(int(k.replace('GS_A_', '').replace('GS_P_', '')))
                        except:#it means it's empty now
                            print('Everything already exists, exiting function...')
                            return

        else:
            print('no keys found')
            dsets['OG'] = f.create_dataset('OG', shape=(N,)+shape, dtype=dtype)
            og = True
            
        assert len(names) != 0, "Data already exists in the specified file:"

        for name in names:
            dsets[name] = f.create_dataset(name, shape=(N,)+shape, dtype=dtype)

        for ind in tqdm(range(N)):
            if og:
                file = files[ind]
                img = np.array(Image.open(file))
                if len(np.array(img).shape) > 2:
                    img = np.mean(img, axis=-1)

                if img.shape != shape:
                    img = smartResize(img, shape)
                img *= mul
                dsets['OG'][ind] = np.round(img).astype(dtype)
            else:
                img = f['OG'][ind].astype(np.float32)
                img /= img.max()
            
            slms, amps = method(img.astype(np.float32), Ks)
            for slm, amp, k in zip(slms, amps, Ks):
                slm = np.mod(slm, 2*np.pi)
                slm = normalize_minmax(slm).numpy()*mul
                dsets["{}_P_{}".format(dset_name, str(k))][ind] = np.round(slm).astype(dtype)
                if not phase_only:
                    amp = normalize_minmax(amp).numpy()*mul
                    dsets["{}_A_{}".format(dset_name, str(k))][ind] = np.round(amp).astype(dtype)

@tf.function
def normalize_minmax(img):
    img = tf.cast(img, tf.float32)
    img -= tf.reduce_min(img, axis=[0, 1], keepdims=True)
    img /= tf.reduce_max(img, axis=[0, 1], keepdims=True)
    return img


@tf.function
def gs(img, Ks):
    phi = tf.random.uniform(img.shape) * np.pi
    slm_phi = tf.random.uniform(img.shape)
    img_cf = tf.complex(tf.random.uniform(img.shape), 0.)
    slm_solutions = []
    amps = []
    
    img = normalize_minmax(img)
    for k in range(Ks[-1]):
        img_cf = tf.complex(img, 0.) * tf.math.exp(tf.complex(0., phi))
        slm_cf = tf.signal.ifft2d(tf.signal.ifftshift(img_cf))
        slm_phi = tf.math.angle(slm_cf)
        slm_cf = tf.math.exp(tf.complex(0., slm_phi))
        img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
        phi = tf.math.angle(img_cf)
        if k in Ks[:-1]:
            slm_solutions.append(normalize_minmax(slm_phi))
            amps.append(tf.math.abs(img_cf))
    slm_solutions.append(slm_phi)
    amps.append(normalize_minmax(tf.math.abs(img_cf)))
    return slm_solutions, amps


def lens(phi_slm):
    slm_cf = tf.math.exp(tf.complex(0., phi_slm))
    img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
    return tf.math.abs(img_cf)


@tf.function
def __normalize_minmax(img):
    img -= tf.reduce_min(tf.cast(img, tf.float32), axis=[0, 1], keepdims=True)
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


def accuracy(y_true, y_pred):
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
        return accuracy(tf.square(img), tf.square(amp))

    for i in range(Ks[-1]+1):
        opt.minimize(loss, var_list=[phi_slm])
        if i in Ks:
            amps.append(loss_(phi_slm).numpy())
            slms.append(phi_slm.numpy())
    return slms, amps



def load_scores_fromCSV(path):
    files = glob(path+"*.csv")
    main_df = pd.read_csv(path+"Main.csv")
    
    for file in files:
        if 'Main' not in file:
            name = os.path.split(file.replace('.csv', ''))[1]
            main_df[name] = np.array(pd.read_csv(file))
    return main_df


def tf_msssim(ref, cgh):
    return tf.image.ssim_multiscale(ref, cgh, 255)


def tf_ssim(ref, cgh):
    return tf.image.ssim(ref[...,np.newaxis], cgh[...,np.newaxis], 255)
    

def brisque(img):
    return brisque_.score(Image.fromarray(img))


def accuracy_batch(y_true, y_pred):
    denom = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis = [1, 2])*tf.reduce_sum(tf.pow(y_true, 2), axis = [1, 2]))
    return 1 - (tf.reduce_sum(y_pred * y_true, axis = [1, 2])+1)/(denom+1)


#%%
























































from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, SpatialDropout2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import HDF5Matrix
import h5py as h5
from tensorflow_addons.layers import WeightNormalization
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


p = "Documents/ModularDeepCGH/image.jpeg"
img = np.mean(np.array(Image.open(p).resize((512,512))), axis=-1).astype(np.float32)
img -= img.min()
img /= img.max()
img = img[np.newaxis, ..., np.newaxis]
imgs = np.concatenate([img for _ in range(64)])

#%%
def accuracy(y_true, y_pred):
    denom = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis=[1, 2, 3])*tf.reduce_sum(tf.pow(y_true, 2), axis=[1, 2, 3])) + 1
    return 1 - tf.reduce_mean((tf.reduce_sum((y_pred) * y_true, axis = [1, 2, 3]))/(denom), axis = 0)

def interleave(x):
    return tf.nn.space_to_depth(input = x,
                               block_size = 8,
                               data_format = 'NHWC')


def deinterleave(x):
    return tf.nn.depth_to_space(input = x,
                               block_size = 8,
                               data_format = 'NHWC')


def __cbn(ten, n_kernels, act_func):
    x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(ten)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(x1)
    x1 = BatchNormalization()(x1)
    return x1 


def __cc(ten, n_kernels, act_func):
    x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(ten)
    x1 = Conv2D(n_kernels, (3, 3), activation = act_func, padding='same')(x1)
    return x1


def __ifft_AmPh(x):
    '''
    Input is Amp x[1] and Phase x[0]. Spits out the angle of ifft.
    '''
    img = tf.dtypes.complex(tf.squeeze(x[0], axis=-1), 0.) * tf.math.exp(tf.dtypes.complex(0., tf.squeeze(x[1], axis=-1)))
    img = tf.signal.ifftshift(img, axes = [1, 2])
    fft = tf.signal.ifft2d(img)
    phase = tf.expand_dims(tf.math.angle(fft), axis=-1)
    return phase

def __phi_slm(phi_slm):
    i_phi_slm = tf.dtypes.complex(np.float32(0.), tf.squeeze(phi_slm, axis=-1))
    return tf.math.exp(i_phi_slm)


def __forward(phi_slm):
    cf_slm = __phi_slm(phi_slm)
    fft = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(cf_slm, axes = [1, 2])), axes = [1, 2])
    return tf.cast(tf.expand_dims(tf.abs(tf.pow(fft, 2)), axis=-1), dtype=tf.dtypes.float32)


def unet(shape):
    n_kernels = [32, 64, 128, 256]
    inp = Input(shape=shape, name='target')
    act_func = 'relu'
    x1_1 = Lambda(interleave, name='Interleave')(inp)
    # Block 1
    x1 = __cbn(x1_1, n_kernels[0], act_func)
    x2 = MaxPooling2D((2, 2), padding='same')(x1)
    # Block 2
    x2 = __cbn(x2, n_kernels[1], act_func)
    encoded = MaxPooling2D((2, 2), padding='same')(x2)
    # Bottleneck
    encoded = __cc(encoded, n_kernels[2], act_func)
    #
    x3 = UpSampling2D(2)(encoded)
    x3 = Concatenate()([x3, x2])
    x3 = __cc(x3, n_kernels[1], act_func)
    #
    x4 = UpSampling2D(2)(x3)
    x4 = Concatenate()([x4, x1])
    x4 = __cc(x4, n_kernels[0], act_func)
    #
    x4 = __cc(x4, n_kernels[1], act_func)
    x4 = Concatenate()([x4, x1_1])
    #
    phi_0_ = Conv2D(8**2, (3, 3), activation=None, padding='same')(x4)
    phi_0 = Lambda(deinterleave, name='phi_0')(phi_0_)
    amp_0_ = Conv2D(8**2, (3, 3), activation='relu', padding='same')(x4)
    amp_0 = Lambda(deinterleave, name='amp_0')(amp_0_)
    
    phi_slm = Lambda(__ifft_AmPh, name='phi_slm')([amp_0, phi_0])
    
    recon = Lambda(__forward, name='out')(phi_slm)    
    
    return Model(inp, recon)

#%%
model = unet((512,512,1))

model.compile(optimizer = 'adam',
             loss = accuracy,
             experimental_run_tf_function = False)

model.optimizer.lr = 0.0001
#%%
model.fit(img, img,
          epochs = 200,
          batch_size = 1,
          verbose = 1)

#%%
recon = model.predict(img)
plt.imshow(np.squeeze(img))
plt.show()
plt.imshow(np.squeeze(recon))
plt.show()




































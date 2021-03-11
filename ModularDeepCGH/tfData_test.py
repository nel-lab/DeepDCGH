import tensorflow as tf
import numpy as np
from glob import glob
from utils import gs, loadImageasGray, smartDType
from sgd_holography import novocgh
from mos_estimator import MOS_Estimator
import matplotlib.pyplot as plt

N_tr = 5
N_te = 5
image_root = './samples'
models_root = './models'
image_size = (512, 512)
dtype = np.uint16
gs_iters = np.array(list(range(1, 40, 10)))
num_gs = len(gs_iters)
novo_iters = np.array(list(range(0, 200, 50)))
num_novo = len(novo_iters)
files_train = glob(image_root+'/train/*.jpg')[:N_tr]
files_test = glob(image_root+'/val/*.jpg')[:N_tr]
mos_est = MOS_Estimator(models_root)
filename_tr = 'train.tfrecord'
filename_te = 'test.tfrecord'



#%
def normalize(img):
    img -= img.min()
    return img / img.max()

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(img, cgh, method, iters, mos):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'image': _bytes_feature(img),
      'cgh': _bytes_feature(cgh),
      'method': _int64_feature(method),
      'iter': _int64_feature(iters),
      'mos': _float_feature(mos),
  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

#%
methods = np.random.randint(0, 6, (len(filename_tr),))
with tf.io.TFRecordWriter(filename_tr) as writer:
    for file, method in zip(files_train, methods):
        img = loadImageasGray(file, image_size, np.uint16).astype(np.float64)/2**16
        print(img.dtype)
        if method in [0, 1, 2]: # it's GS
            iters = gs_iters[np.random.randint(0, num_gs)]
            _, amps = gs(img, [iters])
        elif method in [3, 4, 5]: # it's NOVO
            iters = novo_iters[np.random.randint(0, num_novo)]
            _, amps = novocgh(img, [iters], lr = 0.1)
            # amp_ = amps[0].numpy().astype(np.float32)
            # plt.imshow(amp_)
            # plt.title('right after novo')
            # plt.show()
            
        amp = normalize(amps[0].numpy().astype(np.float64))
        img = normalize(img.astype(np.float64))
        mos = mos_est.get_mos(img, amp)
        img = smartDType(img, dtype)
        amp = smartDType(amp, dtype)
        plt.imshow(amp)
        plt.title('after smart dtype')
        plt.show()
        writer.write(serialize_example(img.tostring(), amp.tostring(), method, iters, mos))

#%%
plt.imshow(img)
plt.show()

#%%
_, amps = novocgh(img, [10], lr = 0.1)
amp = amps[0].numpy().astype(np.float32)

#%%
plt.imshow(amp)
plt.show()


































#%%
filenames = [filename_tr]
raw_dataset = tf.data.TFRecordDataset(filenames)
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'cgh': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'method': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'iter': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'mos': tf.io.FixedLenFeature([], tf.float32, default_value=0),
}

def _parse_function(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.reshape(tf.io.decode_raw(parsed_features['image'], dtype), image_size)
    cgh = tf.reshape(tf.io.decode_raw(parsed_features['cgh'], dtype), image_size)
    method = parsed_features['method']
    iters = parsed_features['iter']
    mos = parsed_features['mos']
    return img, cgh, method, iters, mos

#%%
parsed_dataset = raw_dataset.map(_parse_function)

#%%
import matplotlib.pyplot as plt
meth = ['GS']*3 + ['NOVO']*3
for i in parsed_dataset.take(5):
    # print(i)
    plt.figure(figsize=(10,10))
    plt.imshow(i[0].numpy())
    plt.show()
    plt.figure(figsize=(10,10))
    plt.imshow(i[1].numpy())
    plt.title('Result for {} at {} iterations, estMOS is {}'.format(meth[i[2]], i[3], i[4]))
    plt.show()

#%%
cgh = i[1].numpy().astype(np.float64)
cgh /= 2**16
plt.imshow(cgh)
plt.show()

#%%














#%%

init = 3700
prof = 1.4
for i in range(1000):
    init *= prof
    if init > 1000000:
        break

print(i, init)

#%%



























































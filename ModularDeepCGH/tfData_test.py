import tensorflow as tf
import numpy as np
import IPython.display as display
from glob import glob
from PIL import Image
from utils import gs, loadImageasGray
from sgd_holography import novocgh
import numpy as np

N_tr = 5
N_te = 5
image_root = 'samples'
image_size = (512, 512)
dtype = np.uint16
files_train = glob(image_root+'/train/*.jpg')[:N_tr]
files_test = glob(image_root+'/val/*.jpg')[:N_tr]

filename_tr = 'train.tfrecord'
filename_te = 'test.tfrecord'

#%%
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

def serialize_example(feature0, feature1):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'image': _bytes_feature(feature0),
      'cgh': _bytes_feature(feature1),
      # 'label': _float_feature(feature1),
  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


#%%
train = np.array([np.mean(np.array(Image.open(file).resize(image_size)), axis=-1) for file in files_train]).astype(np.uint16)
cgh_tr = np.array([np.mean(np.array(Image.open(file).resize(image_size)), axis=-1) for file in files_test]).astype(np.uint16)

#%%
# methods = np.rand
with tf.io.TFRecordWriter(filename_tr) as writer:
    for file in files_train:
        img = loadImageasGray(file, image_size, dtype)
        
        writer.write(serialize_example(feature0.tostring(), feature1.tostring()))

#%%
filenames = [filename_tr]
raw_dataset = tf.data.TFRecordDataset(filenames)
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'cgh': tf.io.FixedLenFeature([], tf.string, default_value=''),
    # 'label': tf.io.FixedLenFeature([], tf.float32, default_value=0),
}

def _parse_function(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    # label = parsed_features['label']
    img = tf.reshape(tf.io.decode_raw(parsed_features['image'], np.uint16), (512,512))
    cgh = tf.reshape(tf.io.decode_raw(parsed_features['cgh'], np.uint16), (512,512))
    return img, cgh

#%%
parsed_dataset = raw_dataset.map(_parse_function)

#%%
import matplotlib.pyplot as plt
for i in parsed_dataset.take(3):
    # print(i)
    plt.imshow(i[0].numpy())
    plt.show()
    plt.imshow(i[1].numpy())
    plt.show()

#%%































































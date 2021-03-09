import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import warnings
from skimage.draw import circle, line_aa
import numpy as np
from tqdm import tqdm
from glob import glob
from utils import loadImageasGray, smartDType
from utils import gs
from sgd_holography import novocgh
#%
class DeepCGH_Datasets(object):
    '''
    Class for the Dataset object used in DeepCGH algorithm.
    Inputs:
        path   string, determines the lcoation that the datasets are going to be stored in
        num_iter   int, determines the number of iterations of the GS algorithm
        input_shape   tuple of shape (height, width)
    Returns:
        Instance of the object
    '''
    def __init__(self, params):
        try:
            assert params['object_type'] in ['Disk', 'Line', 'Dot', 'Natural'], 'Object type not supported'
            if params['object_type'] == 'Natural':
                self.image_path = params['images_path']
                self.files_tr = glob(self.image_path+'train/*.jpg')
                self.files_te = glob(self.image_path+'val/*.jpg')
                assert len(self.files_tr)>5 and len(self.files_te)>5, 'Not enough images found. Maybe file path is incorrect.'
                self.count = 0
            self.path = params['path']
            self.shape = params['shape']
            self.N_tr = params['N_train']
            self.N_te = params['N_test']
            self.object_size = params['object_size']
            self.intensity = params['intensity']
            self.object_count = params['object_count']
            self.name = params['name']
            self.object_type = params['object_type']
            self.centralized = params['centralized']
            self.normalize = params['normalize']
            self.compression = params['compression']
            self.dtype = params['dtype']
        except:
            assert False, 'Not all parameters are provided!'
            
        self.__check_avalability()
        
    
    def __check_avalability(self):
        print('Current working directory is:')
        print(os.getcwd(),'\n')
        if self.object_type == 'Natural':
            self.filename = 'Natural_{}_{}_{}_Split.tfrecord'.format(self.N_tr, self.N_te, str(self.dtype))
        else:
            self.filename = self.object_type + '_SHP{}_N{}_SZ{}_INT{}_Crowd{}_CNT{}_Split.tfrecord'.format(self.shape, 
                                               self.N, 
                                               self.object_size,
                                               self.intensity, 
                                               self.object_count,
                                               self.centralized)
        
        self.absolute_file_path = os.path.join(os.getcwd(), self.path, self.filename)
        if not (os.path.exists(self.absolute_file_path.replace('Split', 'Train')) or os.path.exists(self.absolute_file_path.replace('Split', 'Test'))):
            warnings.warn('File does not exist. New dataset will be generated once getDataset is called.')
            print(self.absolute_file_path)
        else:
            print('Data already exists.')
           
            
    def __get_line(self, shape, start, end):
        img = np.zeros(shape, dtype=np.float32)
        rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
        img[rr, cc] = val * 1
        return img
    
    
    def get_circle(self, shape, radius, location):
        """Creates a single circle.
    
        Parameters
        ----------
        shape : tuple of ints
            Shape of the output image
        radius : int
            Radius of the circle.
        location : tuple of ints
            location (x,y) in the image
    
        Returns
        -------
        img
            a binary 2D image with a circle inside
        rr2, cc2
            the indices for a circle twice the size of the circle. This is will determine where we should not create circles
        """
        img = np.zeros(shape, dtype=np.float32)
        rr, cc = circle(location[0], location[1], radius, shape=img.shape)
        img[rr, cc] = 1
        # get the indices that are forbidden and return it
        rr2, cc2 = circle(location[0], location[1], 2*radius, shape=img.shape)
        return img, rr2, cc2


    def __get_allowables(self, allow_x, allow_y, forbid_x, forbid_y):
        '''
        Remove the coords in forbid_x and forbid_y from the sets of points in
        allow_x and allow_y.
        '''
        for i in forbid_x:
            try:
                allow_x.remove(i)
            except:
                continue
        for i in forbid_y:
            try:
                allow_y.remove(i)
            except:
                continue
        return allow_x, allow_y
    
    
    def __get_randomCenter(self, allow_x, allow_y):
        list_x = list(allow_x)
        list_y = list(allow_y)
        ind_x = np.random.randint(0,len(list_x))
        ind_y = np.random.randint(0,len(list_y))
        return list_x[ind_x], list_y[ind_y]
    
    
    def __get_randomStartEnd(self, shape):
        start = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
        end = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
        return start, end


    #% there shouldn't be any overlap between the two circles 
    def __get_RandDots(self, shape, maxnum = [10, 20]):
        '''
        returns a single sample (2D image) with random dots
        '''
        # number of random lines
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        xs = list(np.random.randint(0, shape[0], (n,)))
        ys = list(np.random.randint(0, shape[1], (n,)))
        
        for x, y in zip(xs, ys):
            image[x, y] = 1
            
        return image

    #% there shouldn't be any overlap between the two circles 
    def __get_RandLines(self, shape, maxnum = [10, 20]):
        '''
        returns a single sample (2D image) with random lines
        '''
        # number of random lines
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        for i in range(n):
            # generate centers
            start, end = self.__get_randomStartEnd(shape)
            
            # get circle
            img = self.__get_line(shape, start, end)
            image += img
        image -= image.min()
        image /= image.max()
        return image
    
    #% there shouldn't be any overlap between the two circles 
    def __get_RandBlobs(self, shape, maxnum = [10,12], radius = 5, intensity = 1):
        '''
        returns a single sample (2D image) with random blobs
        '''
        # random number of blobs to be generated
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        try: # in case the radius of the blobs is variable, get the largest diameter
            r = radius[-1]
        except:
            r = radius
        
        # define sets for storing the values
        allow_x = set(range(shape[0]))
        allow_y = set(range(shape[1]))
        if not self.centralized:
            forbid_x = set(list(range(r)) + list(range(shape[0]-r, shape[0])))
            forbid_y = set(list(range(r)) + list(range(shape[1]-r, shape[1])))
        else:
            forbid_x = set(list(range(r)) + list(range(shape[0]-r, shape[0])) + list(range(shape[0]//6, (5)*shape[0]//6)))
            forbid_y = set(list(range(r)) + list(range(shape[1]-r, shape[1])) + list(range(shape[1]//6, (5)*shape[1]//6)))
        
        allow_x, allow_y = self.__get_allowables(allow_x, allow_y, forbid_x, forbid_y)
        count = 0
        # else
        for i in range(n):
            # generate centers
            x, y = self.__get_randomCenter(allow_x, allow_y)
            
            if isinstance(radius, list):
                r = int(np.random.randint(radius[0], radius[1]))
            else:
                r = radius
            
            if isinstance(intensity, list):
                int_4_this = int(np.random.randint(np.round(intensity[0]*100), np.round(intensity[1]*100)))
                int_4_this /= 100.
            else:
                int_4_this = intensity
            
            # get circle
            img, xs, ys = self.get_circle(shape, r, (x,y))
            allow_x, allow_y = self.__get_allowables(allow_x, allow_y, set(xs), set(ys))
            image += img * int_4_this
            count += 1
            if len(allow_x) == 0 or len(allow_y) == 0:
                break
        return image
    
    
    def coord2image(self, coords):
        num_planes = self.shape[-1]
        
        sample = np.zeros(self.shape)
        
        for plane in range(num_planes):
            canvas = np.zeros(self.shape[:-1], dtype=np.float32)
        
            for i in range(coords.shape[-1]):
                img, _, __ = self.get_circle(self.shape[:-1], self.object_size, [coords[0, i], coords[1, i]])
                canvas += img.astype(np.float32)
            
            sample[:, :, plane] = (canvas>0)*1.
            
            if (num_planes > 1) and (plane != 0 and self.normalize == True):
                sample[:, :, plane] *= np.sqrt(np.sum(sample[:, :, 0]**2)/np.sum(sample[:, :, plane]**2))
            
        sample -= sample.min()
        sample /= sample.max()
        
        return np.expand_dims(sample, axis = 0)
    
    
    def get_randSample(self, train_size):
        
        num_planes = self.shape[-1]
        
        sample = np.zeros(self.shape, dtype=self.dtype)
        
        for plane in range(num_planes):
            if self.object_type == 'Disk':
                img = self.__get_RandBlobs(shape = (self.shape[0], self.shape[1]),
                                           maxnum = self.object_count,
                                           radius = self.object_size,
                                           intensity = self.intensity)
                img = smartDType(img, self.dtype)
            elif self.object_type == 'Line':
                img = self.__get_RandLines((self.shape[0], self.shape[1]),
                                           maxnum = self.object_count)
                img = smartDType(img, self.dtype)
            elif self.object_type == 'Dot':
                img = self.__get_RandDots(shape = (self.shape[0], self.shape[1]),
                                          maxnum = self.object_count)
                img = smartDType(img, self.dtype)
            else:
                if self.count < train_size:
                    file = self.files_tr[self.count]
                else:
                    file = self.files_te[self.count-int(train_size)]
                self.count += 1
                img = loadImageasGray(file, self.shape[:-1], self.dtype)

            sample[:, :, plane] = img
            
            # if (num_planes > 1) and (plane != 0 and self.normalize == True):
            #     sample[:, :, plane] *= np.sqrt(np.sum(sample[:, :, 0]**2)/np.sum(sample[:, :, plane]**2))
        
        # sample -= sample.min()
        # sample /= sample.max()
        
        return sample
    
    
    def __bytes_feature(self, value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
    
    
    def __int64_feature(self, value):
      return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    
    
    def __generate(self):
        '''
        Creates a dataset of randomly located blobs and stores the data in an TFRecord file. Each sample (3D image) contains
        a randomly determined number of blobs that are randomly located in individual planes.
        Inputs:
            filename : str
                path to the dataset file
            N: int
                determines the number of samples in the dataset
            fraction : float
                determines the fraction of N that is used as "train". The rest will be the "test" data
            shape: (int, int)
                tuple of integers, shape of the 2D planes
            maxnum: int
                determines the max number of blobs
            radius: int
                determines the radius of the blobs
            intensity : float or [float, float]
                intensity of the blobs. If a scalar, it's a binary blob. If a list, first element is the min intensity and
                second one os the max intensity.
            normalize : bool
                flag that determines whether the 3D data is normalized for fixed energy from plane to plane
    
        Outputs:
            aa:
    
            out_dataset:
                numpy.ndarray. Numpy array with shape (samples, x, y)
        '''
        
#        assert self.shape[-1] > 1, 'Wrong dimensions {}. Number of planes cannot be {}'.format(self.shape, self.shape[-1])
        
        # TODO multiple tfrecord files to store data on. E.g. every 1000 samples in one file
        options = tf.io.TFRecordOptions(compression_type = self.compression)
#        options = None
        a = []
        with tf.io.TFRecordWriter(self.absolute_file_path.replace('Split', 'Train'), options = options) as writer_train:
            with tf.io.TFRecordWriter(self.absolute_file_path.replace('Split', 'Test'), options = options) as writer_test:
                for i in tqdm(np.array(range(self.N_tr+self.N_te)).astype(np.int64)):
                    sample = self.get_randSample(self.N_tr)
                    
                    # label_raw = i.tostring()
                    
                    image_raw = sample.tostring()
                    image_raw2 = (sample//2).tostring()
                    feature = {'sample': self.__bytes_feature(image_raw),
                               # 'mos:': self.__int64_feature(i)},
                               'mos:': self.__bytes_feature(image_raw2)}
                    
                    # 2. Create a tf.train.Features
                    features = tf.train.Features(feature = feature)
                    # 3. Create an example protocol
                    example = tf.train.Example(features = features)
                    # 4. Serialize the Example to string
                    example_to_string = example.SerializeToString()
                    a.append(example_to_string)
                    # 5. Write to TFRecord
                    if i < self.N_tr:
                        writer_train.write(example_to_string)
                    else:
                        writer_test.write(example_to_string)
        return a
        
    
    def getDataset(self, force_delete):
        
        if (os.path.exists(self.absolute_file_path.replace('Split', 'Train')) or os.path.exists(self.absolute_file_path.replace('Split', 'Test'))):
            if force_delete:
                os.remove(self.absolute_file_path.replace('Split', 'Train'))
                os.remove(self.absolute_file_path.replace('Split', 'Test'))
                create_new = True
            else:
                create_new = False
                print('Data already exists')
        else:
            create_new = True
            
        if create_new:
            print('Generating data...')
            folder = os.path.join(os.getcwd(), self.path)
            
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            a = self.__generate()
        self.dataset_paths = [self.absolute_file_path.replace('Split', 'Train'), self.absolute_file_path.replace('Split', 'Test')]
        return a


    def get_tfdatafunc(self, batch_size, epochs, shuffle):
        image_feature_description = {'sample': tf.io.FixedLenFeature([], tf.string, default_value=''),
                                     # 'mos': tf.io.FixedLenFeature([], tf.int64)},
                                     'mos': tf.io.FixedLenFeature([], tf.string, default_value='')}
        
        
        def __parse_image_function(example_proto):
            parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
            img = tf.reshape(tf.io.decode_raw(parsed_features['sample'], self.dtype), self.shape)
            lbl = tf.reshape(tf.io.decode_raw(parsed_features['mos'], self.dtype), self.shape)
            # lbl = parsed_features['mos']
            # print(lbl)
            # lbl = tf.io.decode_raw(parsed_features['mos'], tf.int64)
            return {'target':img}, {'mos':lbl}
        
        def val_func():
            validation = tf.data.TFRecordDataset([self.dataset_paths[1]],
                                  compression_type='GZIP',
                                  buffer_size=None,
                                  num_parallel_reads=2).map(__parse_image_function).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            
            return validation
        
        def train_func():
            train = tf.data.TFRecordDataset([self.dataset_paths[0]],
                                  compression_type='GZIP',
                                  buffer_size=None,
                                  num_parallel_reads=2).map(__parse_image_function).repeat(epochs).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            return train#.shuffle(shuffle)
        return train_func, val_func


#%%
data = {
        'images_path' : 'samples/',
        'path' : 'DeepCGH_Datasets/Natural',
        'shape' : (512, 512, 1),
        'object_type' : 'Natural',
        'dtype' : np.uint16,
        'object_size' : 10,
        'object_count' : [27, 48],
        'intensity' : [0.2, 1],
        'normalize' : True,
        'centralized' : False,
        'N_train' : 10,
        'N_test' : 5,
        'compression' : 'GZIP',
        'name' : 'target',
        }
dset = DeepCGH_Datasets(data)

#%%
a = dset.getDataset(False)

#%%
print(dset.dataset_paths)

#%%
train_func, val_func = dset.get_tfdatafunc(4, 2, 4)

#%%
train_data = train_func()#.as_numpy_iterator()

#%%
for i in train_data:
    print(i)

#%%
import matplotlib.pyplot as plt

#%%
for samp in train_data:
    print(samp[0]['target'].numpy().shape)
    plt.imshow(samp[0]['target'].numpy()[0, :,:,0])
    plt.title(str(samp[1]['mos'].numpy()[0]))
    plt.show()
    plt.imshow(samp[0]['target'].numpy()[1, :,:,0])
    plt.title(str(samp[1]['mos'].numpy()[1]))
    plt.show()
    plt.imshow(samp[0]['target'].numpy()[2, :,:,0])
    plt.title(str(samp[1]['mos'].numpy()[2]))
    plt.show()
    plt.imshow(samp[0]['target'].numpy()[3, :,:,0])
    plt.title(str(samp[1]['mos'].numpy()[3]))
    plt.show()


#%%
a = samp[0]['target'].numpy()[3,:,:,0]

#%%
a = np.array([1,2,3])
s = []

for aa in np.array(range(3)):
    s.append(aa.tostring())

#%%
raw_dataset = tf.data.TFRecordDataset(dset.dataset_paths[0])

#%%
for raw_record in raw_dataset:
  print(repr(raw_record))

#%%






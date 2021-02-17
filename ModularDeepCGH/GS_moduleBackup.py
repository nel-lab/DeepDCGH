class GS(object):

    def __init__(self, Ks, shape, batch_size):

        self.Ks = Ks
        self.shape = shape
        self.batch_size = batch_size
        self.input_name = 'target'
        self.input_queue = Queue(maxsize=4)
        self.output_queue = Queue(maxsize=4)
        self.__initGS()

    def __initGS(self):
        model_fn = self.__get_model_fn()

        self.estimator = tf.estimator.Estimator(model_fn,
                                                model_dir="")

        self.__start_thread()

    def __start_thread(self):
        self.prediction_thread = Thread(target=self.__predict_from_queue, daemon=True)
        self.prediction_thread.start()

    def __generate_from_queue(self):
        '''
        A generator with infinite loop to fetch one smaple from the queue
        Returns:
            one sample!
        '''
        while True:
            yield self.input_queue.get()

    def __predict_from_queue(self):
        '''
        Once the input queue has something to offer to the estimator, the
        estimator makes the predictions and the outputs of that prediction are
        fed into the output queue.
        '''
        for i in self.estimator.predict(input_fn=self.__queued_predict_input_fn,
                                        yield_single_examples=False):
            self.output_queue.put(i)

    def get_hologram(self, inputs):
        '''
        Return the hologram using the GS algorithm with num_iter iterations.
        Inputs:
            inputs   numpy ndarray, the two dimentional target image
        Returns:
            hologram as a numpy ndarray
        '''
        features = {}
        if not isinstance(self.input_name, str):
            for key, val in zip(self.input_name, inputs):
                features[key] = val
        else:
            features = {self.input_name: inputs}
        self.input_queue.put(features)
        predictions = self.output_queue.get()

        return predictions  #[self.output_name]

    def __queued_predict_input_fn(self):
        '''
        Input function that returns a tensorflow Dataset from a generator.
        Returns:
            a tensorflow dataset
        '''
        # Fetch the inputs from the input queue
        type_dict = {}
        shape_dict = {}

        if not isinstance(self.input_name, str):
            for key in self.input_name:
                type_dict[key] = tf.float32
                shape_dict[key] = (None,) + self.shape
        else:
            type_dict = {self.input_name: tf.float32}
            shape_dict = {self.input_name: (None,) + self.shape}

        dataset = tf.data.Dataset.from_generator(self.__generate_from_queue,
                                                 output_types=type_dict,
                                                 output_shapes=shape_dict)
        return dataset

    def __get_model_fn(self):
        Ks = self.Ks
        shape = self.shape
        bs = self.batch_size

        @tf.function
        def __normalize_minmax(img):
            img -= tf.reduce_min(tf.cast(img, tf.float32), axis=[1, 2])
            img /= tf.reduce_max(img, axis=[1, 2])
            return img

        # @tf.function
        def __gs(img):
            rand_phi = tf.squeeze(tf.random.uniform((bs,)+shape), axis = -1)
            slm_solutions = []
            amps = []
            phis = []
            img = __normalize_minmax(tf.squeeze(img, axis = -1))
            for k in tqdm(range(Ks[-1])):
                if k != 0:
                    img_cf = tf.complex(img, 0.) * tf.math.exp(tf.complex(0., phi))
                else:
                    img_cf = tf.complex(img, 0.) * tf.math.exp(tf.complex(0., rand_phi))

                slm_cf = tf.signal.ifft2d(tf.signal.ifftshift(img_cf))
                slm_phi = tf.math.angle(slm_cf)
                slm_cf = tf.math.exp(tf.complex(0., slm_phi))
                img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
                phi = tf.math.angle(img_cf)
                if k == Ks[:-1]:
                    slm_solutions.append(tf.expand_dims(slm_phi, axis = -1))
                    phis.append(tf.expand_dims(phi, axis = -1))
                    amps.append(tf.expand_dims(tf.math.abs(img_cf), axis = -1))
            slm_solutions.append(slm_phi)
            phis.append(phi)
            amps.append(__normalize_minmax(tf.math.abs(img_cf)))
            return tf.concat(slm_solutions, axis=-1), tf.concat(amps, axis=-1), tf.concat(phis, axis=-1)

        def model_fn(features, labels, mode):
            # phi_slm = __gs(features['target'])
            phi_slm, amps, phis = __gs(features['target'])
            return tf.estimator.EstimatorSpec(mode, predictions=[phi_slm, amps, phis])
            # return tf.estimator.EstimatorSpec(mode, predictions = phi_slm)
        return model_fn


#%%
import numpy as np
import h5py as h5
from PIL import Image
from tqdm import tqdm
import os
from glob import glob
import tensorflow as tf
# from gs import GS
p = '/storage1/datasets/natural_images/COCO/test2017/000000193437.jpg'
img = Image.open(p)
img = np.mean(np.array(img), axis=-1, keepdims=True)

#%%

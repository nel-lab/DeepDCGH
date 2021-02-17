import tensorflow as tf

@tf.function
def normalize_minmax(img):
    img -= tf.reduce_min(tf.cast(img, tf.float32), axis=[0, 1], keepdims=True)
    img /= tf.reduce_max(img, axis=[0, 1], keepdims=True)
    return img


@tf.function
def gs(img, Ks):
    rand_phi = tf.random.uniform(img.shape)
    slm_solutions = []
    amps = []

    # print("right before first minmax image size is {}".format(img.shape))
    img = normalize_minmax(img)
    for k in range(Ks[-1]):
        if k != 0:
            img_cf = tf.complex(img, 0.) * tf.math.exp(tf.complex(0., phi))
        else:
            img_cf = tf.complex(img, 0.) * tf.math.exp(tf.complex(0., rand_phi))

        slm_cf = tf.signal.ifft2d(tf.signal.ifftshift(img_cf))
        slm_phi = tf.math.angle(slm_cf)
        slm_cf = tf.math.exp(tf.complex(0., slm_phi))
        img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
        phi = tf.math.angle(img_cf)
        if k in Ks[:-1]:
            slm_solutions.append(slm_phi)
            amps.append(tf.math.abs(img_cf))
    slm_solutions.append(slm_phi)

    amps.append(normalize_minmax(tf.math.abs(img_cf)))
    return slm_solutions, amps
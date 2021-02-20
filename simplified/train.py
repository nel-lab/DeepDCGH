import tensorflow as tf
from Network import Generator, Discriminator
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from skimage import io
import numpy as np
from numpy import array
from skimage.transform import resize
import os
from tqdm import tqdm

np.random.seed(10)
image_shape = (384,384,3)

vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
vgg19.trainable = False
for l in vgg19.layers:
    l.trainable = False
loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
loss_model.trainable = False

def vgg_loss(y_true, y_pred):
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def get_gan_network(discriminator, shape, generator, optimizer):
    # discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories


def load_data_from_dirs(dirs, ext, size):
    files = []
    count = 0
    for d in dirs:
        for f in os.listdir(d): 
            if f.endswith(ext):
                files.append(resize(io.imread(os.path.join(d,f)), size))
                count = count + 1
                if count == 1001:
                    break
    return files


def load_data(directory, ext):

    files = load_data_from_dirs(load_path(directory), ext, image_shape)
    return files


files = load_data("./data", ".jpg")
x_train = files[:-100]
x_test = files[-100:]

print("data loaded")


def hr_images(images):
    images_hr = array(images)
    images_hr -= images_hr.min()
    images_hr /= images_hr.max()    
    return images_hr

def lr_images(images_real , downscale):
    images = []
    for img in  range(len(images_real)):
        images.append(resize(images_real[img], [images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale]))
    images_lr = array(images)
    return images_lr

def preprocess_HR(x):
    return np.divide(x.astype(np.float32), .5) - np.ones_like(x,dtype=np.float32)


def deprocess_HR(x):
    input_data = (x + 1) * .5
    return input_data.astype(np.uint8)

def normalize(input_data):
    return (input_data.astype(np.float32) - .5)/.5
    
def denormalize(input_data):
    input_data -= input_data.min()
    input_data /= input_data.max()
    return input_data

x_train_hr = hr_images(x_train)
# x_train_hr = normalize(x_train_hr)

x_train_lr = lr_images(x_train, 4)
# x_train_lr = normalize(x_train_lr)


x_test_hr = hr_images(x_test)
# x_test_hr = normalize(x_test_hr)

x_test_lr = lr_images(x_test, 4)
# x_test_lr = normalize(x_test_lr)



##### test
# plt.imshow(x_train_hr[0])
# plt.show()
# plt.hist(x_train_hr[0].reshape(-1))
# plt.show()
# plt.imshow(x_train_lr[0])
# plt.show()
# plt.hist(image_batch_hr[0].reshape(-1))
# plt.show()


######################



print("data processed")

def plot_generated_images(epoch,generator, examples=3 , dim=(2, 3), figsize=(15, 5)):
    rand_nums = np.random.randint(0, x_test_hr.shape[0], size=examples)
    image_batch_hr = x_test_hr[rand_nums]
    # image_batch_hr = denormalize(image_batch_hr)
    image_batch_lr = x_test_lr[rand_nums]
    gen_img = generator.predict(image_batch_lr)
    generated_image = normalize(gen_img)
    # generated_image = denormalize(gen_img)
    # image_batch_lr = denormalize(image_batch_lr)
    
    #generated_image = deprocess_HR(generator.predict(image_batch_lr))
    
    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[1], interpolation='nearest')
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[1], interpolation='nearest')
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[1], interpolation='nearest')
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 4)
    plt.hist(image_batch_lr[1].reshape(-1))
        
    plt.subplot(dim[0], dim[1], 5)
    plt.hist(generated_image[1].reshape(-1))
    
    plt.subplot(dim[0], dim[1], 6)
    plt.hist(image_batch_hr[1].reshape(-1))
    
    plt.tight_layout()
    plt.savefig('output/gan_generated_image_epoch_%d.png' % epoch)
    

def train(epochs=1, batch_size=4):
# epochs = 20000
# batch_size = 4
    downscale_factor = 4
    
    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    
    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()
    print('gen and dis created')
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=vgg_loss, optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)
    
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, 3)
    gan = get_gan_network(discriminator, shape, generator, adam)
    
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            generated_images_sr = generator.predict(image_batch_lr)
    
            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2
            
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
    
            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])
            
        print("Loss HR , Loss LR, Loss GAN")
        print(d_loss_real, d_loss_fake, loss_gan)
    
        if e == 1 or e % 5 == 0:
            plot_generated_images(e, generator)
        if e % 20 == 0:
            generator.save('./output/gen_model%d.h5' % e)
            discriminator.save('./output/dis_model%d.h5' % e)
            gan.save('./output/gan_model%d.h5' % e)

train(20000,4)



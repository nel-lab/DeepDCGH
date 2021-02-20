import PySimpleGUI as sg
import os
from glob import glob
from PIL import Image, ImageTk
import io
import numpy as np
import h5py as h5
import pandas as pd
max_qual = 9
q = max_qual//2
count = 0

keypad_mapping = {'KP_End:87' : 1,'KP_Down:88' : 2,'KP_Next:89' : 3,'KP_Left:83' : 4,
                    'KP_Begin:84' : 5,'KP_Right:85' : 6,'KP_Home:79' : 7,'KP_Up:80' : 8,
                    'KP_Prior:81' : 9,'KP_Insert:90' : 0,'1:10' : 1,'2:10' : 2,
                    '3:10' : 3,'4:10' : 4,'5:10' : 5,'6:10' : 6,'7:10' : 7,
                    '8:10' : 8,'9:10' : 9,'0:19' : 0,'1' : 1,'2' : 2,'3' : 3,
                    '4' : 4,'5' : 5,'6' : 6,'7' : 7,'8' : 8,'9' : 9,'0' : 0}

def fft(phase):
    phase = phase.astype('float32')
    phase /= phase.max()
    phase *= np.pi
    slm_cf = np.exp(1j*phase)
    img = np.abs(np.fft.fftshift(np.fft.fft2(slm_cf)))
    img -= img.min()
    img /= img.max()
    img *= 255
    img = img.astype('uint8')
    return img
    

def get_img_data(data, maxsize=(1024, 1024), first=False, do_fft = False):
    if do_fft:
        data = fft(data)
    img = Image.fromarray(data)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

def update_indexes(img_idx, method_idx, up_or_down, max_img, max_method):
    if up_or_down:
        if method_idx == max_method-1:
            if img_idx == max_img-1:
                pass
            else:
                img_idx += 1
                method_idx = 0
        else:
            method_idx += 1
    else:
        if method_idx == 0:
            if img_idx == 0:
                pass
            else:
                img_idx -= 1
                method_idx = max_method-1
        else:
            method_idx -= 1
    
    return img_idx, method_idx


def popup(message):
    file = sg.popup_get_file(message, default_path='')
    if not file:
        sg.popup_cancel('Cancelling')
        raise SystemExit()
    return file


def getORcheckDF(file, num_images, methods):
    df_file = file.replace('.h5', '.csv')
    if os.path.isfile(df_file):
        df = pd.read_csv(df_file)
        
        for i in range(num_images):
            for j, method in enumerate(methods):
                if df[method][i] == 123:
                    df[method][i] = q
                    return df, i, j
    else:
        df = pd.DataFrame(data = np.ones((num_images, len(methods)), dtype=np.uint8)*123, columns = methods)
        df[methods[0]][0] = q
        return df, 0, 0


# get file address
file = popup('Choose the file that contains images.')


with h5.File(file, 'r') as f:
    
    ogs = f['OG']
    num_images = ogs.shape[0]
    method_names = list(f.keys())
    method_names.remove('OG')
    method_names = sorted(method_names)

    dataFrame, img_index, method_index = getORcheckDF(file, num_images, method_names)

    progressbar_elem = sg.ProgressBar(num_images*len(method_names), orientation='h', size=(80, 20), key='progbar')
    OG_elem = sg.Image(data = get_img_data(ogs[img_index], first = True, do_fft = False), size = (512,512))
    CGH_elem = sg.Image(data = get_img_data(f[method_names[method_index]][img_index], first = True, do_fft = True), size = (512,512))
    slider_elem = sg.Slider(range = (0, max_qual),
                            default_value=q,
                            key = '_SLIDER_',
                            orientation = 'h',
                            enable_events = True,
                            disable_number_display = False,
                            size = (80, 20))

    layout = [[progressbar_elem],
              [sg.T('Original Image', size = (85, 1)), sg.T('Rate this hologram', size = (20, 1))],
              [OG_elem, CGH_elem],
              [sg.T('Low Quality'), slider_elem, sg.T('Best Quality')],
              [sg.Button('Prev', size=(8, 2)),
               sg.In(max_qual//2, key = '_Score_'),
               sg.Button('Next', size=(8, 2))]]
    
    window = sg.Window('Image Scoring Software', layout, return_keyboard_events=True,
                       location=(0, 0), use_default_focus=False)

    while True:
        # read the form
        event, values = window.read()
        # perform button and keyboard operations
        if event == sg.WIN_CLOSED:
            img_index, method_index = update_indexes(img_index,
                                                 method_index,
                                                 True,#up
                                                 max_img = num_images,
                                                 max_method = len(method_names))
            if dataFrame[method_names[method_index]][img_index] == 123:
                img_index, method_index = update_indexes(img_index,
                                                     method_index,
                                                     False,#up
                                                     max_img = num_images,
                                                     max_method = len(method_names))
                dataFrame[method_names[method_index]][img_index] = 123
            break
        
        elif event in list(keypad_mapping.keys()):
            dataFrame[method_names[method_index]][img_index] = keypad_mapping[event]
            if img_index == num_images-1 and method_index == len(method_names)-1:
                break
            img_index, method_index = update_indexes(img_index,
                                                 method_index,
                                                 True,#up
                                                 max_img = num_images,
                                                 max_method = len(method_names))
            if dataFrame[method_names[method_index]][img_index] == 123:
                dataFrame[method_names[method_index]][img_index] = q
        
        elif (event == 'Next' or 'Right:' in event) and 'KP' not in event:
            dataFrame[method_names[method_index]][img_index] = values['_Score_']
            if img_index == num_images-1 and method_index == len(method_names)-1:
                break
            img_index, method_index = update_indexes(img_index,
                                                 method_index,
                                                 True,#up
                                                 max_img = num_images,
                                                 max_method = len(method_names))
            if dataFrame[method_names[method_index]][img_index] == 123:
                dataFrame[method_names[method_index]][img_index] = q
            
        elif (event == 'Prev' or 'Left:' in event) and 'KP' not in event:
            dataFrame[method_names[method_index]][img_index] = values['_Score_']
            img_index, method_index = update_indexes(img_index,
                                                 method_index,
                                                 False,#up
                                                 max_img = num_images,
                                                 max_method = len(method_names))
        
        elif event == '_SLIDER_':
            dataFrame[method_names[method_index]][img_index] = values['_SLIDER_']
            
        elif 'Down' in event and 'KP' not in event:
            if dataFrame[method_names[method_index]][img_index] == 0:
                dataFrame[method_names[method_index]][img_index] = 0
            else:
                dataFrame[method_names[method_index]][img_index] -= 1
        
        elif 'Up' in event and 'KP' not in event:
            if dataFrame[method_names[method_index]][img_index] == max_qual:
                dataFrame[method_names[method_index]][img_index] = max_qual
            else:
                dataFrame[method_names[method_index]][img_index] += 1
        
        window['_SLIDER_'].update(dataFrame[method_names[method_index]][img_index])
        window['_Score_'].update(dataFrame[method_names[method_index]][img_index])
        count = (img_index) * len(method_names) + method_index
        window['progbar'].update(count)
        # update window with new image
        OG_elem.update(data=get_img_data(ogs[img_index], first=False))
        CGH_elem.update(data=get_img_data(f[method_names[method_index]][img_index], first=False))
        if count % len(method_names) == 0:
            dataFrame.to_csv(file.replace('.h5', '.csv'), index=False)
            
dataFrame.to_csv(file.replace('.h5', '.csv'), index=False)
window.close()

#%%
# import numpy as np
# import pandas as pd
# rates = np.random.rand(*(20, 12))
# cols = ['a{}'.format(i) for i in range(12)]
# df = pd.DataFrame(data = rates, columns=cols)

# #%%
# df.to_csv('test_dataframe.csv')

# #%%
# df = pd.DataFrame()
# for i in range(10):
#     df = df.append(pd.DataFrame(data=np.random.rand(1,12), columns=cols))

# #%%
# a=pd.DataFrame(data=np.random.rand(1,12), columns=cols)

# #%%
# start = 9800
# perc = []
# for i in range(3*4+5*12):
#     perc.append(4000 *(1.02**i)/(start))
#     start += 4000 *(1.02**i)


# #%%
# from PIL import Image
# import numpy as np
# import h5py as h5
# from glob import glob
# import matplotlib.pyplot as plt

# files = glob('./images/*')

# #%
# imgs = np.concatenate([np.mean(np.array(Image.open(img).resize((512, 512))), axis=-1)[np.newaxis, ...] for img in files], axis=0)

# #%
# imgs -= np.min(imgs, axis=(1,2), keepdims=True)
# imgs /= np.max(imgs, axis=(1,2), keepdims=True)
# all_imgs = [imgs[np.newaxis, ...]]
# for i in range(3):
#     all_imgs.append((imgs + (i/6)*np.random.rand(5, 512, 512))[np.newaxis, ...])
# imgs = np.concatenate(all_imgs, axis=0)
# imgs -= np.min(imgs, axis=(2,3), keepdims=True)
# imgs /= np.max(imgs, axis=(2,3), keepdims=True)
# imgs *= 255
# imgs = np.round(imgs).astype('uint8')

# #%%
# for i in range(5):
#     plt.imshow(imgs[0, i], cmap='gray')
#     plt.show()
#     plt.imshow(imgs[1, i], cmap='gray')
#     plt.show()
#     plt.imshow(imgs[2, i], cmap='gray')
#     plt.show()
#     plt.imshow(imgs[3, i], cmap='gray')
#     plt.show()

# #%%
# dsets = []
# with h5.File('test.h5', 'a') as f:
#     og = f.create_dataset('OG', data=imgs[0])
#     for i in range(3):
#         dsets.append(f.create_dataset('img_set{}'.format(i), data=imgs[1+i]))

# #%%
# with h5.File(file, 'r') as f:
#     method_names = list(f.keys())
#     method_names.remove('OG')
#     method_names = sorted(method_names)




#%%
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py as h5
import numpy as np
file = '/home/hoss/Documents/COCO2017_Size512_N50.h5'
cghs = []
names = []
with h5.File(file, 'r') as f:
    images = f['OG'][:]
    for k in f.keys():
        if k != 'OG':
            names.append(k)
            cghs.append(f[k][:])

#%%
def fft(phase):
    slm_cf = tf.math.exp(tf.complex(0., phase))
    img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
    img = tf.math.abs(img_cf)
    return img

#%%
for cgh, name in zip(cghs, names):
    # plt.imshow(cgh[0])
    # plt.title(name)
    # plt.show()
    if '_A_' not in name:
        img = fft(cgh[0].astype(np.float32)).numpy()
        plt.imshow(img)
        plt.show()
        plt.hist(img.reshape(-1), 100)
        plt.show()

#%%
@tf.function
def normalize_minmax(img):
    img -= tf.reduce_min(tf.cast(img, tf.float32), axis=[0, 1], keepdims=True)
    img /= tf.reduce_max(img, axis=[0, 1], keepdims=True)
    return img


@tf.function
def gs(img, Ks):
    phi = tf.random.uniform(img.shape) * np.pi
    slm_solutions = []
    amps = []

    # print("right before first minmax image size is {}".format(img.shape))
    img = normalize_minmax(img)
    for k in range(Ks[-1]):
        img_cf = tf.complex(img, 0.) * tf.math.exp(tf.complex(0., phi))

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

#%%
img = imgages[0]
slm, amp = gs()





































































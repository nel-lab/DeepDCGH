import PySimpleGUI as sg
import os
from glob import glob
from PIL import Image, ImageTk
import io
import numpy as np
import h5py as h5
import pandas as pd
import random

max_qual = 9
q = max_qual//2

def get_randomOrder(file, max_imgs, max_methods, methods):
    order_file = file.replace('.h5', '_idx.csv')
    qual_file = file.replace('.h5', '.csv')
    if os.path.isfile(order_file) and os.path.isfile(qual_file):
        quals = pd.read_csv(qual_file)
        order = pd.read_csv(order_file)
        for i in range(max_imgs*max_methods):
            if not order['done'][i]:
                return quals, order, i
    else:
        quals = pd.DataFrame(data = np.ones((max_imgs, max_methods), dtype=np.uint8)*123, columns = methods)
        quals[methods[0]][0] = q
        order = np.zeros((max_imgs*max_methods, 3), dtype=np.uint16)
        count = 0
        for i in range(max_imgs):
            for j in range(max_methods):
                order[count, 0] = i
                order[count, 1] = j
                count+=1
        order = list(order)
        random.shuffle(order)
        order = pd.DataFrame(data = order, columns=['image', 'method', 'done'])
        return quals, order, 0

keypad_mapping = {'KP_End:87' : 1,'KP_Down:88' : 2,'KP_Next:89' : 3,'KP_Left:83' : 4,
                    'KP_Begin:84' : 5,'KP_Right:85' : 6,'KP_Home:79' : 7,'KP_Up:80' : 8,
                    'KP_Prior:81' : 9,'KP_Insert:90' : 0,'1:10' : 1,'2:10' : 2,
                    '3:10' : 3,'4:10' : 4,'5:10' : 5,'6:10' : 6,'7:10' : 7,
                    '8:10' : 8,'9:10' : 9,'0:19' : 0,'1' : 1,'2' : 2,'3' : 3,
                    '4' : 4,'5' : 5,'6' : 6,'7' : 7,'8' : 8,'9' : 9,'0' : 0}
    

def get_img_data(data, maxsize=(1024, 1024), first=False):
    img = Image.fromarray(data)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)


def popup(message):
    file = sg.popup_get_file(message, default_path='')
    if not file:
        sg.popup_cancel('Cancelling')
        raise SystemExit()
    return file


# get file address
file = popup('Choose the file that contains images.')


with h5.File(file, 'r') as f:
    count = 0
    ogs = f['OG']
    num_images = ogs.shape[0]
    method_names = list(f.keys())
    method_names.remove('OG')
    method_names = sorted(method_names)
    
    quals, order, count = get_randomOrder(file, num_images, len(method_names), method_names)
    img_index = order['image'][count]
    method_index = order['method'][count]
    quals[method_names[method_index]][img_index] = q
    
    progressbar_elem = sg.ProgressBar(num_images*len(method_names), orientation='h', size=(80, 20), key='progbar')
    OG_elem = sg.Image(data = get_img_data(ogs[img_index], first = True), size = (512,512))
    CGH_elem = sg.Image(data = get_img_data(f[method_names[method_index]][img_index], first = True), size = (512,512))
    slider_elem = sg.Slider(range = (0, max_qual),
                            default_value=q,
                            key = '_SLIDER_',
                            orientation = 'h',
                            enable_events = True,
                            disable_number_display = False,
                            size = (80, 20))

    layout = [[sg.T('Total Progress:'), progressbar_elem],
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
            img_index = order['image'][count+1]
            method_index = order['method'][count+1]
            if quals[method_names[method_index]][img_index] == 123 and order['order'][count+1]==0:
                img_index = order['image'][count]
                method_index = order['method'][count]
                quals[method_names[method_index]][img_index] = 123
            break
        
        elif event in list(keypad_mapping.keys()):
            quals[method_names[method_index]][img_index] = keypad_mapping[event]
            order['done'][count] = 1
            # are we done yet?
            if count == len(method_names)*num_images-1:
                break
            count += 1
            img_index = order['image'][count]
            method_index = order['method'][count]
            if quals[method_names[method_index]][img_index] == 123:
                quals[method_names[method_index]][img_index] = q
            
        
        elif (event == 'Next' or 'Right:' in event) and 'KP' not in event:
            quals[method_names[method_index]][img_index] = values['_Score_']
            order['done'][count] = 1
            if count == len(method_names)*num_images-1:
                break
            count += 1
            img_index = order['image'][count]
            method_index = order['method'][count]
            if quals[method_names[method_index]][img_index] == 123:
                quals[method_names[method_index]][img_index] = q
            
        elif (event == 'Prev' or 'Left:' in event) and 'KP' not in event:
            quals[method_names[method_index]][img_index] = values['_Score_']
            order['done'][count] = 1
            count -= 1
            img_index = order['image'][count]
            method_index = order['method'][count]
        
        elif event == '_SLIDER_':
            quals[method_names[method_index]][img_index] = values['_SLIDER_']
            
        elif 'Down' in event and 'KP' not in event:
            if quals[method_names[method_index]][img_index] == 0:
                quals[method_names[method_index]][img_index] = 0
            else:
                quals[method_names[method_index]][img_index] -= 1
        
        elif 'Up' in event and 'KP' not in event:
            if quals[method_names[method_index]][img_index] == max_qual:
                quals[method_names[method_index]][img_index] = max_qual
            else:
                quals[method_names[method_index]][img_index] += 1
        
        window['_SLIDER_'].update(quals[method_names[method_index]][img_index])
        window['_Score_'].update(quals[method_names[method_index]][img_index])
        window['progbar'].update(count)
        # update window with new image
        OG_elem.update(data=get_img_data(ogs[img_index], first=False))
        CGH_elem.update(data=get_img_data(f[method_names[method_index]][img_index], first=False))
        if count % 5 == 0:
            quals.to_csv(file.replace('.h5', '.csv'), index=False)
            order.to_csv(file.replace('.h5', '_idx.csv'), index=False)
            
quals.to_csv(file.replace('.h5', '.csv'), index=False)
order.to_csv(file.replace('.h5', '_idx.csv'), index=False)
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




# #%%
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import h5py as h5
# import numpy as np
# file = '/home/hoss/Documents/datasets/COCO2017_Size512_N1000.h5'
# cghs = []
# names = []
# with h5.File(file, 'r') as f:
#     images = f['OG'][:]
#     for k in f.keys():
#         if k != 'OG':
#             names.append(k)
#             cghs.append(f[k][:])
# img = images[9].astype(np.float32)
# img /= img.max()

# #%%
# plt.imshow(img)
# plt.show()

# #%%
# def fft(phase):
#     slm_cf = tf.math.exp(tf.complex(0., phase))
#     img_cf = tf.signal.fftshift(tf.signal.fft2d(slm_cf))
#     img = tf.math.abs(img_cf)
#     return img

# #%%
# set
# for cgh, name in zip(cghs, names):
#     # plt.imshow(cgh[0])
#     # plt.title(name)
#     # plt.show()
#     phi = cgh[9].copy()
#     # plt.hist(phi.reshape(-1), 100)
#     # plt.title(name)
#     # plt.show()
#     if '_A_' not in name:
#         phi = (phi.astype(np.float32) / phi.max())*2*np.pi
#         img_ = fft(phi).numpy()
#         plt.figure(figsize=(10,10))
#         plt.imshow(img_)
#         plt.title(name)
#         plt.savefig(name+'1.png')
#         plt.show()
        
#         # test: make it 8bit
#         img_ -= img_.min()
#         img_ /= img_.max()
#         img_ *= 2**8
#         img_ = np.round(img_).astype(np.uint8)
        
#         plt.figure(figsize=(10,10))
#         plt.imshow(img_)
#         plt.title(name)
#         plt.savefig(name+'2.png')
#         plt.show()

# #%%
# final_slms = []
# slms, amps = novocgh(img, [5,400, 1000], lr=1)
# for slm, amp, k in zip(slms, amps, [5,10,100]):
#     final_slms.append(normalize_minmax(slm).numpy()*2*np.pi)

# #%%
# for it, phiii in enumerate(final_slms):
#     phi = (phiii.astype(np.float32) / phi.max())*2*np.pi
#     img_ = fft(phi).numpy()
#     plt.imshow(img_)
#     plt.title(it)
#     plt.show()
    
# #%%
# plt.imshow(img)
# plt.show()

# plt.imshow(amps[2])
# plt.show()

# plt.imshow(img_)
# plt.show()

# #%%





















































import PySimpleGUI as sg
import os
from glob import glob
from PIL import Image, ImageTk
import io
import numpy as np
import h5py as h5
import pandas as pd
max_qual = 20
q = max_qual//2
count = 0

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


def random_image():
    return np.round(np.random.rand(512, 512, 3)*255).astype(np.uint8)

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
                    return df, i, j
    else:
        df = pd.DataFrame(data = np.ones((num_images, len(methods)), dtype=np.uint8)*123, columns = methods)
        return df, 0, 0


# get file address
file = popup('Choose the file that contains images.')


with h5.File(file, 'r') as f:
    
    ogs = f['OG']
    num_images = ogs.shape[0]
    method_names = list(f.keys())
    method_names.remove('OG')
    method_names = sorted(method_names)

    progressbar_elem = sg.ProgressBar(num_images*len(method_names), orientation='h', size=(100, 20), key='progbar')
    OG_elem = sg.Image(data = get_img_data(ogs[0], first = True), size = (512,512))
    CGH_elem = sg.Image(data = get_img_data(f[method_names[0]][0], first = True), size = (512,512))
    slider_elem = sg.Slider(range = (0, max_qual),
                            default_value=max_qual//2,
                            key = '_SLIDER_',
                            orientation = 'h',
                            enable_events = True,
                            disable_number_display = False,
                            size = (150,20))

    layout = [[progressbar_elem],
              [sg.T('Original Image', size = (100, 1)), sg.T('Rate this hologram', size = (20, 1))],
              [OG_elem, CGH_elem],
              [sg.T('Low Quality'), slider_elem, sg.T('Best Quality')],
              [sg.Button('Prev', size=(8, 2)),
               sg.In(max_qual//2, key = '_Score_'),
               sg.Button('Next', size=(8, 2))]]
    
    window = sg.Window('Image Scoring Software', layout, return_keyboard_events=True,
                       location=(0, 0), use_default_focus=False)

    dataFrame, img_index, method_index = getORcheckDF(file, num_images, method_names)
    
    while True:        
        # read the form
        event, values = window.read()
        # perform button and keyboard operations
        if event == sg.WIN_CLOSED:
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
            dataFrame[method_names[method_index]][img_index] = q
            count += 1 
        
        elif (event == 'Next' or 'Right:' in event) and 'KP' not in event:
            dataFrame[method_names[method_index]][img_index] = values['_Score_']
            if img_index == num_images-1 and method_index == len(method_names)-1:
                break
            img_index, method_index = update_indexes(img_index,
                                                 method_index,
                                                 True,#up
                                                 max_img = num_images,
                                                 max_method = len(method_names))
            dataFrame[method_names[method_index]][img_index] = q
            count += 1
            
        elif (event == 'Prev' or 'Left:' in event) and 'KP' not in event:
            dataFrame[method_names[method_index]][img_index] = values['_Score_']
            img_index, method_index = update_indexes(img_index,
                                                 method_index,
                                                 False,#up
                                                 max_img = num_images,
                                                 max_method = len(method_names))
            count -= 1
        
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
        window['progbar'].update(count)
        # update window with new image
        OG_elem.update(data=get_img_data(ogs[img_index], first=False))
        CGH_elem.update(data=get_img_data(f[method_names[method_index]][img_index], first=False))
        if count % len(method_names) == 0:
            dataFrame.to_csv(file.replace('.h5', '.csv'))
            
dataFrame.to_csv(file.replace('.h5', '.csv'))
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








































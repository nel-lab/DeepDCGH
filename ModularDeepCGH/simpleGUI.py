import PySimpleGUI as sg
import os
from glob import glob
from PIL import Image, ImageTk
import io
import numpy as np
import time
import h5py as h5
import pandas as pd
max_qual = 9
keypad_mapping = {
                    'KP_End:87' : 1,
                    'KP_Down:88' : 2,
                    'KP_Next:89' : 3,
                    'KP_Left:83' : 4,
                    'KP_Begin:84' : 5,
                    'KP_Right:85' : 6,
                    'KP_Home:79' : 7,
                    'KP_Up:80' : 8,
                    'KP_Prior:81' : 9,
                    'KP_Insert:90' : 0,
                    '1:10' : 1,
                    '2:10' : 2,
                    '3:10' : 3,
                    '4:10' : 4,
                    '5:10' : 5,
                    '6:10' : 6,
                    '7:10' : 7,
                    '8:10' : 8,
                    '9:10' : 9,
                    '0:19' : 0,
                    '1' : 1,
                    '2' : 2,
                    '3' : 3,
                    '4' : 4,
                    '5' : 5,
                    '6' : 6,
                    '7' : 7,
                    '8' : 8,
                    '9' : 9,
                    '0' : 0}

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
        if method_idx == max_method:
            if img_idx == max_img:
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
                method_idx = max_method
        else:
            method_idx -= 1
        
    
    return img_idx, method_idx


file = sg.popup_get_file('Image file to open', default_path='')
if not file:
    sg.popup_cancel('Cancelling')
    raise SystemExit()

# asser #TODO popupt file.lower().endswith('h5'), 'Wrong file format!'


progressbar_elem = sg.ProgressBar(100, orientation='h', size=(100, 20), key='progbar')
OG_elem = sg.Image(data = get_img_data(random_image(), first = True), size = (512,512))
CGH_elem = sg.Image(data = get_img_data(random_image(), first = True), size = (512,512))
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
          [slider_elem],
          [sg.Button('Prev', size=(8, 2)),
           sg.In(max_qual//2, key = '_Score_'),
           sg.Button('Next', size=(8, 2))]]

window = sg.Window('Image Score Recording', layout, return_keyboard_events=True,
                   location=(0, 0), use_default_focus=False)

img_index = 0
method_index = 0

with h5.File(file, 'r') as f:
    keys = sorted(list(f.keys()).remove('OG'))
    qs = [max_qual//2]*len(keys)
    df = pd.DataFrame()
    
    while True:
        
        og = f['OG'][img_index, :, :]
        max_img = f['OG'].shape[0]
        cgh = f[keys[method_index]][img_index, :, :]
        
        # read the form
        event, values = window.read()
        # perform button and keyboard operations
        if event == sg.WIN_CLOSED:
            break
        
        elif event in list(keypad_mapping.keys()):
            qs[method_index] = keypad_mapping[event]
            img_idx, method_idx = update_indexes(img_index,
                                                 method_index,
                                                 True,#up
                                                 max_img = max_img,
                                                 max_method = len(keys))
        
        if (event == 'Next' or 'Right:' in event) and 'KP' not in event:
            All_Qs[i] = values['_Score_']
            i += 1
            if i >= total_num:
                break # TODO popup
        elif (event == 'Prev' or 'Left:' in event) and 'KP' not in event:
            if i ==0:
                i = 0
            else:
                i -= 1
        
        
        elif event == '_SLIDER_' or ('Down' in event and 'KP' not in event):
            if event == '_SLIDER_':
                All_Qs[i] = values['_SLIDER_']
            else:
                if All_Qs[i] == 0:
                    All_Qs[i] = 0
                else:
                    All_Qs[i] -= 1
        
        elif event == '_SLIDER_' or ('Up' in event and 'KP' not in event):
            if event == '_SLIDER_':
                All_Qs[i] = values['_SLIDER_']
            else:
                if All_Qs[i] == max_qual:
                    All_Qs[i] = max_qual
                else:
                    All_Qs[i] += 1
                
        # window['labels'].update(event)
        
        window['_SLIDER_'].update(All_Qs[i])
        window['_Score_'].update(All_Qs[i])
        window['progbar'].update(i)
        
        img = imgs[i]
        cgh = cghs[i]
        # update window with new image
        OG_elem.update(data=get_img_data(img, first=True))
        CGH_elem.update(data=get_img_data(cgh, first=True))

window.close()
print(All_Qs)


#%%
import numpy as np
import pandas as pd
rates = np.random.rand(*(20, 12))
cols = ['a{}'.format(i) for i in range(12)]
df = pd.DataFrame(data = rates, columns=cols)

#%%
df.to_csv('test_dataframe.csv')

#%%
df = pd.DataFrame()
for i in range(10):
    df = df.append(pd.DataFrame(data=np.random.rand(1,12), columns=cols))

#%%
a=pd.DataFrame(data=np.random.rand(1,12), columns=cols)

#%%
start = 9800
perc = []
for i in range(3*4+5*12):
    perc.append(4000 *(1.02**i)/(start))
    start += 4000 *(1.02**i)

















import PySimpleGUI as sg
import os
from glob import glob
from PIL import Image, ImageTk
import io
import numpy as np
import h5py as h5
import pandas as pd

qual_range = [1,5]

q = 3

def get_quals(file, max_imgs):
    qual_file = file.replace('.h5', '.csv')
    
    if os.path.isfile(qual_file):
        quals = pd.read_csv(qual_file)
        for i in range(len(quals)):
            if quals['Score'][i]==123:
                return quals, i
    else:
        quals = pd.DataFrame(data = np.ones((max_imgs, 1), dtype=np.uint8)*123, columns = ['Score'])
        return quals, 0

keypad_mapping = {'KP_End:87' : 1,'KP_Down:88' : 2,'KP_Next:89' : 3,'KP_Left:83' : 4,
                    'KP_Begin:84' : 5,'KP_Right:85' : 6,'KP_Home:79' : 7,'KP_Up:80' : 8,
                    'KP_Prior:81' : 9,'KP_Insert:90' : 0,'1:10' : 1,'2:10' : 2,
                    '3:10' : 3,'4:10' : 4,'5:10' : 5,'6:10' : 6,'7:10' : 7,
                    '8:10' : 8,'9:10' : 9,'0:19' : 0,'1' : 1,'2' : 2,'3' : 3,
                    '4' : 4,'5' : 5,'6' : 6,'7' : 7,'8' : 8,'9' : 9,'0' : 0}

def get_img_data(data, first=False, size = (1024, 768)):
    img = Image.fromarray(data)
    img = img.resize(size)
    img.thumbnail(size)
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
if '_half_' in file:
    size = (512//2, 385//2)
elif '_quart_' in file:
    size = (512//4, 385//4)
else:
    size = (512, 385)

with h5.File(file, 'r') as f:
    ogs = f['OG']
    cghs = f['Amplitudes']
    num_images = ogs.shape[0]
    method_names = list(f.keys())
    method_names.remove('OG')
    method_names = sorted(method_names)
    
    quals, count = get_quals(file, num_images)
    quals['Score'][count] = q
    
    progressbar_elem = sg.ProgressBar(num_images, orientation='h', size=(80, 20), key='progbar')
    OG_elem = sg.Image(data = get_img_data(ogs[count], first = True, size = size), size = size)
    CGH_elem = sg.Image(data = get_img_data(cghs[count], first = True, size = size), size = size)
    slider_elem = sg.Slider(range = (qual_range[0], qual_range[1]),
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
               sg.In(q, key = '_Score_'),
               sg.Button('Next', size=(8, 2))]]
    
    window = sg.Window('Image Scoring Software', layout, return_keyboard_events=True,
                       location=(0, 0), use_default_focus=False)
    
    while True:
        # read the form
        event, values = window.read()
        # perform button and keyboard operations 
        if event == sg.WIN_CLOSED:
            if quals['Score'][count+1] == 123:
                quals['Score'][count] = 123
            break
        
        
        elif event in list(keypad_mapping.keys()):
            quals['Score'][count] = keypad_mapping[event]
            # are we done yet?
            if count == num_images-1:
                break
            count += 1
            if quals['Score'][count] == 123:
                quals['Score'][count] = q
            
        
        elif (event == 'Next' or 'Right:' in event) and 'KP' not in event:
            quals['Score'][count] = values['_Score_']
            if count == num_images-1:
                break
            count += 1
            if quals['Score'][count] == 123:
                quals['Score'][count] = q
            
        elif (event == 'Prev' or 'Left:' in event) and 'KP' not in event:
            quals['Score'][count] = values['_Score_']
            count -= 1
        
        elif event == '_SLIDER_':
            quals['Score'][count] = values['_SLIDER_']
            
        elif 'Down' in event and 'KP' not in event:
            if quals['Score'][count] == qual_range[0]:
                quals['Score'][count] = qual_range[0]
            else:
                quals['Score'][count] -= 1
        
        elif 'Up' in event and 'KP' not in event:
            if quals['Score'][count] == qual_range[1]:
                quals['Score'][count] = qual_range[1]
            else:
                quals['Score'][count] += 1
        
        window['_SLIDER_'].update(quals['Score'][count])
        window['_Score_'].update(quals['Score'][count])
        window['progbar'].update(count)
        # Update window with new image 
        OG_elem.update(data=get_img_data(ogs[count], first=False, size = size))
        CGH_elem.update(data=get_img_data(cghs[count], first=False, size = size))
        if count % 5 == 0:
            quals.to_csv(file.replace('.h5', '.csv'), index=False)

quals.to_csv(file.replace('.h5', '.csv'), index=False)
window.close()

#%% To delete inappropriate content
# import h5py as h5
# import numpy as np
# with h5.File('/home/hoss/Documents/datasets/koniq/KonIQ_384_512_1488.h5', 'r') as f:
#     ogs = f['OG'][:]
#     cghs = f['Amplitudes'][:]

# #%%
# ogs = np.delete(ogs , 59 , axis = 0)
# cghs = np.delete(cghs , 59 , axis = 0)

# #%%
# with h5.File('IQA_GUI.h5' , 'w') as f:
#     dset = f.create_dataset('OG', data=ogs[:1000])
#     dset1 = f.create_dataset('Amplitudes', data=cghs[:1000])



























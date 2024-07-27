#%%
import cv2
import os
import pandas as pd
import numpy as np
import glob
import scipy.io

name = 'SMBk'
# Read calibration file and worm ID
framecali = '../frame_calibration_checked/framecali_wen1119.xlsx'
id_data = pd.read_excel(framecali, sheet_name=None)
keys = list(id_data.keys())
assert len(keys) == 1, 'More than one sheet in the excel file'
id_data = id_data[keys[0]]
# second column is the start index of each trial; third column is the end index
trial_start_ind = list(id_data.iloc[:,1].values.astype(int))
trial_end_ind = list(id_data.iloc[:,2].values.astype(int))
# worm id strings are non-empty elements in the first column
id_ind = np.where(~id_data.iloc[:,0].isnull())[0]
file_root_ls = list(id_data.iloc[id_ind,0].values)
if len(file_root_ls) >0 and file_root_ls[0][0] == "'":
    file_root_ls = [file_root_ls[i][1:-5] for i in range(len(file_root_ls))]
num_worm = len(id_ind)
# print(num_worm, id_ind)
print('number of worms: ', num_worm)
print('worm start trial index: ', id_ind)

#%%
# video_folder_dir = ['/Volumes/Lenovo/Pinjie/lpj paper videos 205/N2/20190811/A20190811','/Volumes/Lenovo/Pinjie/lpj paper videos 205/N2/20190812']
# video_folder_dir = ['/Volumes/Lenovo/Pinjie/wen1037_flp22__minisog/video']
# video_folder_dir = ['/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1123 exp 20211117','/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1123exp 20211126']
# video_folder_dir = ['/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20200928','/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20200929','/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20201024','/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20201025']
video_folder_dir = folder_ls = ["/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1119 exp 20210805/", "/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1119 exp 20210812/"]
save_path = '/Volumes/Lenovo/Worm_Clips/SMBk_clips/forward_clips/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
video_dir = [glob.glob(video_folder_dir[i]+'/*.avi') for i in range(len(video_folder_dir))]
video_dir = [item for sublist in video_dir for item in sublist]
video_name_ls = [video_dir[i].split('/')[-1][:-4] for i in range(len(video_dir))]
curvature = np.load('../curvature/{}.npy'.format(name),allow_pickle=True)
print(video_name_ls)
#%%
# extract frames from videos

if not os.path.exists(save_path):
    os.makedirs(save_path)
last_video = None
N_trial = len(trial_start_ind)
id_ind_appended = np.append(id_ind, N_trial)
curv_window = 500
for worm_i in range(num_worm):
    file_root = file_root_ls[worm_i]
    # find the position of file_root str identical to the video name
    matching = [file_root == video_name_ls[j] for j in range(len(video_name_ls))]
    if np.sum(matching) == 0:
        print('No matching for file root: ', file_root)
        continue
    else:
        video_ind = np.where(matching)[0][0]
    video_path = video_dir[video_ind]
    cap = cv2.VideoCapture(video_path)
    start_end_ls = [[trial_start_ind[i], trial_end_ind[i]] for i in range(id_ind_appended[worm_i], id_ind_appended[worm_i+1])]
    print('worm: ', file_root_ls[worm_i])
    print('video path: ', video_path)
    print('start end list: ', start_end_ls)
    print('-------------------')
    # clip the video into trials
    ret, frame = cap.read()
    h, w, _ = frame.shape # 512 384
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_ls = [cv2.VideoWriter(os.path.join(save_path, 'trail_{}.mp4'.format(i+id_ind[worm_i])), fourcc, 30.0, (w,h+120)) for i in range(len(start_end_ls))]
    f = 0
    kt_ls = [curvature[i,0][:15].mean(axis=0) for i in range(id_ind_appended[worm_i], id_ind_appended[worm_i+1])]
    kt_zero_ls = [(-kt_ls[i].min()).astype(np.uint16)*4 + h for i in range(len(kt_ls))]
    kt_int_ls = [(kt_ls[i] - kt_ls[i].min()).astype(np.uint16)*4 + h for i in range(len(kt_ls))]
    while ret:
        for i in range(len(start_end_ls)):
            if f >= start_end_ls[i][0] and f < start_end_ls[i][1]:
                # print frame number on the frame and write in 
                #pad the frame with 100 pixels on the bottom
                frame = cv2.copyMakeBorder(frame, 0, 120, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                t = f - start_end_ls[i][0]
                plot_curv = kt_int_ls[i][max(0,t-curv_window//2):min(len(kt_int_ls[i]),t+curv_window//2)]
                x = np.arange(250-min(t,curv_window//2), 250+min(curv_window//2, len(kt_int_ls[i])-t))
                y = np.stack((x,plot_curv), axis=1)
                cv2.polylines(frame, [y], False, (0,255,0), 2)
                #red point at the current frame
                cv2.circle(frame, (250,kt_int_ls[i][t]), 4, (0,0,255), -1)
                # plot the horizontal dashed line
                cv2.line(frame, (50,kt_zero_ls[i]), (w-50, kt_zero_ls[i]), (0,0,100), 1, cv2.LINE_AA)
                cv2.putText(frame, 'frame: {}'.format(f), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                writer_ls[i].write(frame)
                
        ret, frame = cap.read()
        f += 1
    cap.release()
    for i in range(len(start_end_ls)):
        writer_ls[i].release()



# %%

from python import radar
import matplotlib.pyplot as plt
import glob
import os
import imageio
import cv2
import numpy as np
import scipy.io as sio
from skimage import io


Rad_img=True

if Rad_img:
    i=0
    ncols=4
else:
    i=-1
    ncols=3

#scene = 3
data_dir_image_info = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/tiny_foggy/final-rad-info_annotated/' 
data_dir_original = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/tiny_foggy/Navtech_Cartesian_annotated/'
data_dir_sparse = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/tiny_foggy/reconstruct/reshaped_annotated/'
data_dir_prev_info = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/tiny_foggy/Navtech_Cartesianorg_annotated/'


data_path = os.path.join(data_dir_image_info,'*png')
files = sorted(glob.glob(data_path))

for num,images in enumerate(files):
    #if Rad_img==True:
    #    if num<1:
    #        continue
    print(images)
    X_image_info = Xorig = cv2.imread(images)#, cv2.IMREAD_GRAYSCALE)

    original_file = data_dir_original + images[100:]

    X_original = cv2.imread(original_file)#, cv2.IMREAD_GRAYSCALE)
    
    sparse_file = data_dir_sparse + images[100:]
    print(sparse_file)
    X_sparse = cv2.imread(sparse_file)#, cv2.IMREAD_GRAYSCALE)
    
    if Rad_img:
        prev_file = data_dir_prev_info + images[100:]
        X_info_prev = cv2.imread(prev_file)#, cv2.IMREAD_GRAYSCALE)

    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(20,20))
    if Rad_img:
        axs[i].axis('off')
        full_title = images[100:] + ' Org annotation'# prev Image info'
        axs[i].title.set_text(full_title)
        axs[i].imshow(X_info_prev, cmap='gray', vmin=0, vmax=255)
    full_title = images[100:] + ' Rad-info'
    axs[i+1].axis('off')
    axs[i+1].title.set_text(full_title)
    axs[i+1].imshow(X_image_info, cmap='gray', vmin=0, vmax=255)
    axs[i+2].axis('off')
    axs[i+2].title.set_text('Sparse-baseline')
    axs[i+2].imshow(X_sparse, cmap='gray', vmin=0, vmax=255)
    axs[i+3].axis('off')
    axs[i+3].title.set_text('orig-radar-network')
    axs[i+3].imshow(X_original, cmap='gray', vmin=0, vmax=255)
    #plt.savefig('test.png')
    plt.show()
    #break






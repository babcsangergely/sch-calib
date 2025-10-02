import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.ndimage import rotate
from scipy.signal import savgol_filter
import glob
from skimage.transform import resize
from panorama import outdata, load_vid, loadfile
import pickle
from scipy import ndimage
import sys
import os


vid, framerate, startframe = load_vid(1708, preset="gap1e",d_trigger=[0,502])
cut = vid[:, 220:255, 70:140]




img_array=np.array(np.array(cut[500],dtype="float")-cut[1],dtype="float")

plt.imshow(cut[500])
plt.show()

plt.imshow(img_array)  

plt.show()


dif=0
angle=0
max_val=0
for i in range (180*5):
    rotated = rotate(img_array, i/5, reshape=True)  
    rows=rotated.shape[0]
    row_sums = np.sum(rotated, axis=1)
    counts = np.count_nonzero(rotated, axis=1)
    row_means = np.divide(row_sums, counts, out=np.full_like(row_sums, np.nan, dtype=float), where=counts!=0)


    rmc=row_means[10:-10]
    if (np.max(rmc)-np.min(rmc)>dif):
        dif=np.max(rmc)-np.min(rmc)
        angle=i/5



rotated = rotate(img_array, angle, reshape=True)
plt.imshow(rotated)

plt.show()
print(rotated)
print(angle)
print (dif)

row_sums = np.sum(rotated, axis=1)
counts = np.count_nonzero(rotated, axis=1)
row_means = np.divide(row_sums, counts, out=np.full_like(row_sums, np.nan, dtype=float), where=counts!=0)
rmc=row_means[10:-10]
plt.plot(rmc)
plt.show()

row_means_smooth = savgol_filter(rmc, window_length=9, polyorder=3)
plt.plot(row_means_smooth)
plt.show()



background=np.mean(rmc[:15])


density=np.zeros(len(rmc)+1)
pixel_length=0.11/1000  # pixel size in meters
m= 0.021200478544405908
b=-background*m
for i in reversed(range (len(rmc))):
    density[i-1]=density[i]+ (m * rmc[i] + b) * pixel_length
density=density*1000
plt.plot(density)
plt.ylabel('Density (g/m^3)')
plt.show()


rot_smooth = np.repeat(row_means_smooth[:, np.newaxis], len(row_means_smooth)*2, axis=1)
img_smooth=rotate(rot_smooth,-1*angle, reshape=True)

def center_crop(img, target_shape):
    
    tY, tX = target_shape
    y, x = img.shape
    
    startx = x//2 - (tX//2)
    starty = y//2 - (tY//2)
    
    return img[starty:starty+tY, startx:startx+tX]

img_smooth_cropped = center_crop(img_smooth, img_array.shape)

plt.imshow(img_array, cmap="gray", alpha=0.8)
plt.imshow(img_smooth_cropped, cmap="inferno", alpha=0.4)
plt.show()

plt.imshow(img_smooth_cropped, cmap="gray")
plt.show()
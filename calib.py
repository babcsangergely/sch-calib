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


vid, framerate, startframe = load_vid(1708, preset="gap1e",d_trigger=[0,1800])
cut = vid[:, 230:260, 70:160]
plt.imshow(np.array(cut[500],dtype="float"),vmin=0,vmax=130)
plt.show()
plt.imshow(np.array(np.array(cut[500],dtype="float")-cut[0],dtype="float"),vmin=-10,vmax=13, cmap="grey")
plt.show()




img_array=np.array(np.array(cut[500],dtype="float")-cut[0],dtype="float")
#img_array[img_array==0]=np.nan
plt.imshow(img_array, cmap='gray')  # 'gray' ensures it's shown correctly
plt.axis('off')  # Hide axes
plt.show()
print(img_array.shape)



dif=0
angle=0
max=0
for i in range (180*5):
    rotated = rotate(img_array, i/5, reshape=True)  
    rows=rotated.shape[0]
    row_sums = np.sum(rotated, axis=1)
    counts = np.count_nonzero(rotated, axis=1)
    row_means = np.divide(row_sums, counts, out=np.full_like(row_sums, np.nan, dtype=float), where=counts!=0)


    
    if (np.max(row_means)-np.min(row_means)>dif):
        dif=np.max(row_means)-np.min(row_means)
        angle=i/5
    if(np.max(row_means)>max):
        mangle=i/5
        max=np.max(row_means)
        line_row = np.argmax(row_means)

print(row_means)

rotated = rotate(img_array, angle, reshape=True)
plt.imshow(rotated, cmap='gray')
plt.axis('off')
plt.show()
print(rotated)
print(angle)
print (dif)
rotated = rotate(img_array, mangle, reshape=True)
plt.imshow(rotated, cmap='gray')
plt.axhline(line_row, color='red')  # draw detected line
plt.axis('off')
plt.show()
print(max)
row_sums = np.sum(rotated, axis=1)
counts = np.count_nonzero(rotated, axis=1)
row_means = np.divide(row_sums, counts, out=np.full_like(row_sums, np.nan, dtype=float), where=counts!=0)
plt.plot(row_means)
plt.show()

row_means_smooth = savgol_filter(row_means, window_length=9, polyorder=3)
plt.plot(row_means_smooth)
plt.show()

background=np.mean(row_means[:19])


density=np.zeros(len(row_means)+1)
pixel_length=0.11/1000  # pixel size in meters
m= 0.021200478544405908
b=-background*m
for i in range (len(row_means)):
    density[i+1]=density[i]+ (m * row_means[i] + b) * pixel_length
plt.plot(density)
plt.ylabel('Density (kg/m^3)')
plt.show()

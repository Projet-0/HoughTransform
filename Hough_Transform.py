#from __future__ import division
#import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
import skimage.io
from skimage.color import rgb2gray
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter


#original_image = cv2.imread('C:/Users/Tanguy travail/Downloads/Pic1.jpeg',1)
#gray_image = cv2.imread('Sample_Input.jpg',0)

#output = original_image.copy()

#Gaussian Blurring of Gray Image
#blur_image = cv2.GaussianBlur(original_image,(3,3),0)

#Using OpenCV Canny Edge detector to detect edges
#edged_image = cv2.Canny(blur_image,75,150)
#cv2.imshow('Edged Image', edged_image)

#print("A")

start_time = time.time()

# original_image = mpimg.imread('/home/pi/Documents/une-ping-pong.jpeg')

original_image = mpimg.imread('C:/Users/Tanguy travail/Downloads/HoughTransform-main/HoughTransform-main/une-ping-pong.jpeg')
R,G,B = original_image[:,:,0],original_image[:,:,1],original_image[:,:,2]
ima = 0.2989*R + 0.5870*G + 0.1140*B
blured = gaussian_filter(ima, sigma=1)

im = feature.canny(blured, sigma=1, low_threshold = 50, high_threshold = 150)

radi = 50
pi = 3.14159

Rmin = 50
Rmax = 150
height,width,j = original_image.shape

#cosl = np.zeros(radi)
#sinl = np.zeros(radi)
#for i in range(radi):
#     cosl[i] = math.cos(2*pi*i/radi)
#    sinl[i] = math.sin(2*pi*i/radi)


def point(arr,radi,x,y,Cmax): # Tranformation d'un point, arr = Accumulation; radi = Nombre d'angle; x,y = Coordonées
    for r in range(Rmin,Rmax,4):
        for i in range(radi):
            teta = 2*pi*i/radi
            #a = int(Rmax + x + r*cosl[i])
            #b = int(Rmax + y + r*sinl[i])
            a = int(Rmax + x + r*math.cos(teta))
            b = int(Rmax + y + r*math.sin(teta))
            arr[a][b][r] += 1
            if arr[a][b][r]>Cmax[3]:
                Cmax = [a,b,r,arr[a][b][r]]
            # Variable global du max
    return Cmax

# def max_2D(arr, height, width):
#     max, x, y, z = 0, 0, 0, 0
#     for i in range(2*Rmax + height):
#         for j in range(2*Rmax + width):
#             for k in range(Rmax):
#                 if arr[i][j][k]>max:
#                     x,y,z = i,j,k
#                     max = arr[i][j][k]
#                     print(max)
#                 if arr[i][j][k]>= 100:
#                     print(i,j,k,arr[i][j][k])
#     return x,y,z,max,height,width

def hough_transform(im): # Transformation d'un image filtre par canny
    height,width = im.shape
    Cmax = [0,0,0,0] # a,b,r,valeur
    tot_array = np.zeros((height+2*Rmax,width+2*Rmax,Rmax))
    k = 0
    for i in range(height):
        for j in range(width):
            if im[i][j]!=0:
                k +=1
                Cmax = point(tot_array,radi,i,j,Cmax)
    print(Cmax,k)
    # return max_2D(tot_array, height, width, radi)

def hough_transform2(im):
    height,width = im.shape
    Cmax = [0,0,0,0] # a,b,r,valeur
    tot_array = np.zeros((height+2*Rmax,width+2*Rmax,Rmax))
    k = 0
    for i in range(0,height,4):
        for j in range(0,width,4):
            if im[i][j]!=0:
                k +=1
                Cmax = point(tot_array,radi,i,j,Cmax)
    print(Cmax,k)
    return(Cmax,k)

Cmax,k = hough_transform2(im)
az = np.full((5,5),255)
print(height, width)
print(original_image[0:2][0:2][0])
for i in range(Cmax[2]):
    if (Cmax[0]-Rmax+i+2 < width) and (Cmax[0]-Rmax+i-2 >= 0) and (Cmax[1]-Rmax+2 < height) and (Cmax[1]-Rmax-2 >= 0):
        print(Cmax[0]-Rmax+i+2)
        original_image[Cmax[0]-Rmax+i-2:Cmax[0]-Rmax+i+2][Cmax[1]-Rmax-2:Cmax[1]-Rmax+2][0] = az
    if (Cmax[0]-Rmax-i+2 < width) and (Cmax[0]-Rmax-i-2 >= 0) and (Cmax[1]-Rmax+2 < height) and (Cmax[1]-Rmax-2 >= 0):
        original_image[Cmax[0]-Rmax-i-2:Cmax[0]-Rmax-i+2][Cmax[1]-Rmax-2:Cmax[1]-Rmax+2][0] = az
    if (Cmax[0]-Rmax+2 < width) and (Cmax[0]-Rmax-2 >= 0) and (Cmax[1]-Rmax+i+2 < height) and (Cmax[1]-Rmax+i-2 >= 0):
        original_image[Cmax[0]-Rmax-2:Cmax[0]-Rmax+2][Cmax[1]-Rmax+i-2:Cmax[1]-Rmax+i+2][0] = az
    if (Cmax[0]-Rmax+2 < width) and (Cmax[0]-Rmax-2 >= 0) and (Cmax[1]-Rmax-i+2 < height) and (Cmax[1]-Rmax-i-2 >= 0):
        print(Cmax[0]-Rmax-2)
        print(Cmax[1]-Rmax-i-2)
        original_image[Cmax[0]-Rmax-2:Cmax[0]-Rmax+2,Cmax[1]-Rmax-i-2:Cmax[1]-Rmax-i+2][0] = az

for i in range(500):
    teta = 2*pi*i/500
    if int(Cmax[0]-Rmax+Cmax[2]*math.cos(teta))-2 >= 0 and int(Cmax[0]-Rmax+Cmax[2]*math.cos(teta))+2 < width and int(Cmax[1]-Rmax+Cmax[2]*math.sin(teta))-2 >= 0 and int(Cmax[1]-Rmax+Cmax[2]*math.sin(teta))+2 < height:
        original_image[int(Cmax[0]-Rmax+Cmax[2]*math.cos(teta))-2:int(Cmax[0]-Rmax+Cmax[2]*math.cos(teta))+2][int(Cmax[1]-Rmax+Cmax[2]*math.sin(teta))-2:int(Cmax[1]-Rmax+Cmax[2]*math.sin(teta))+2][2] = 255
#cv2.imshow('Identification', original_image)


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax[0].imshow(original_image)
ax[0].set_title('Image originalle', fontsize=20)

ax[1].imshow(blured, cmap='gray')
ax[1].set_title('Image après blur', fontsize=20)

ax[2].imshow(im, cmap='gray')
ax[2].set_title('Canny', fontsize=20)

for a in ax:
    a.axis('off')

fig.tight_layout()



#cv2.imshow('Edged Image', edged_image)

end_time = time.time()
time_taken = end_time - start_time
print ('Time taken for execution',time_taken)

plt.show()

#cv2.waitKey(0)
#cv2.destroyAllWindows()


from __future__ import division
import cv2
import numpy as np
import time
import math


original_image = cv2.imread('C:/Users/Tanguy travail/Downloads/Pic1.jpeg',1)
#gray_image = cv2.imread('Sample_Input.jpg',0)

output = original_image.copy()

#Gaussian Blurring of Gray Image
blur_image = cv2.GaussianBlur(original_image,(3,3),0)

#Using OpenCV Canny Edge detector to detect edges
edged_image = cv2.Canny(blur_image,75,150)
cv2.imshow('Edged Image', edged_image)

print("A")

start_time = time.time()

pi = 3.14159

radi = 50

Rmin = 50
Rmax = 500

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
            if im[i][j]==255:
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
            if im[i][j]==255:
                k +=1
                Cmax = point(tot_array,radi,i,j,Cmax)
    print(Cmax,k)
    return(Cmax,k)

Cmax,k = hough_transform2(edged_image)
for i in range(Cmax[2]):
    original_image[Cmax[0]-Rmax+i][Cmax[1]-Rmax][2] = 255
    original_image[Cmax[0]-Rmax-i][Cmax[1]-Rmax][2] = 255
    original_image[Cmax[0]-Rmax][Cmax[1]-Rmax+i][2] = 255
    original_image[Cmax[0]-Rmax][Cmax[1]-Rmax-i][2] = 255
cv2.imshow('Identification', original_image)



cv2.imshow('Edged Image', edged_image)

end_time = time.time()
time_taken = end_time - start_time
print ('Time taken for execution',time_taken)

cv2.waitKey(0)
cv2.destroyAllWindows()
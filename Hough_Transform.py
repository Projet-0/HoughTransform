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
from picamera import PiCamera
from time import sleep

link = '/home/pi/Desktop/image0.jpg'

start_time = time.time()

k = 0

# list1 = ['/home/pi/Desktop/image',k, '.jpg'] # Changement de type on le passe en liste pour implémenter le numéro de l'image
# str1 = ''.join(list1)  #On repasse en format string pour stocker l'image dans la camera
# print(str1)

camera = PiCamera() #on définit camera comme la caméra de la raspberry
camera.resolution = (1024,768) #définir la résolution on va devoir rester en 16/9

radi = 50
pi = 3.14159

Rmin = 10
Rmax = 150

pasR = 4
pasXY = 4

az = np.full((5,5),255) # Tableau blue de 5*5

def point(arr,radi,x,y,Cmax): # Tranformation d'un point, arr = Accumulation; radi = Nombre d'angle; x,y = Coordonées
    for r in range(Rmin,Rmax,pasR): # Pas
        for i in range(radi):
            teta = 2*pi*i/radi
            a = int(Rmax + x + r*math.cos(teta))
            b = int(Rmax + y + r*math.sin(teta))
            arr[a][b][r] += 1
            if arr[a][b][r]>Cmax[3]:
                Cmax = [a,b,r,arr[a][b][r]]
            # Variable global du max
    return Cmax

def hough_transform(im):
    height,width = im.shape
    Cmax = [0,0,0,0] # a,b,r,valeur
    tot_array = np.zeros((height+2*Rmax,width+2*Rmax,Rmax))
    k = 0
    for i in range(0,height,pasXY): # Pas
        for j in range(0,width,pasXY): #Pas
            if im[i][j]!=0:
                k +=1
                Cmax = point(tot_array,radi,i,j,Cmax)
    return(Cmax,k)

def affichage(Cmax):

    for i in range(Cmax[2]):
        if (Cmax[0]-Rmax+i+2 < width) and (Cmax[0]-Rmax+i-2 >= 0) and (Cmax[1]-Rmax+2 < height) and (Cmax[1]-Rmax-2 >= 0):
            original_image[Cmax[0]-Rmax+i-2:Cmax[0]-Rmax+i+3,Cmax[1]-Rmax-2:Cmax[1]-Rmax+3,0] = az
        if (Cmax[0]-Rmax-i+2 < width) and (Cmax[0]-Rmax-i-2 >= 0) and (Cmax[1]-Rmax+2 < height) and (Cmax[1]-Rmax-2 >= 0):
            original_image[Cmax[0]-Rmax-i-2:Cmax[0]-Rmax-i+3,Cmax[1]-Rmax-2:Cmax[1]-Rmax+3,0] = az
        if (Cmax[0]-Rmax+2 < width) and (Cmax[0]-Rmax-2 >= 0) and (Cmax[1]-Rmax+i+2 < height) and (Cmax[1]-Rmax+i-2 >= 0):
            original_image[Cmax[0]-Rmax-2:Cmax[0]-Rmax+3,Cmax[1]-Rmax+i-2:Cmax[1]-Rmax+i+3,0] = az
        if (Cmax[0]-Rmax+2 < width) and (Cmax[0]-Rmax-2 >= 0) and (Cmax[1]-Rmax-i+2 < height) and (Cmax[1]-Rmax-i-2 >= 0):
            original_image[Cmax[0]-Rmax-2:Cmax[0]-Rmax+3,Cmax[1]-Rmax-i-2:Cmax[1]-Rmax-i+3,0] = az
        
    for i in range(500):
        teta = 2*pi*i/500
        original_image[int(Cmax[0]-Rmax+Cmax[2]*math.cos(teta))][int(Cmax[1]-Rmax+Cmax[2]*math.sin(teta))][0] = 255
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
    
    plt.show()
    
# Fin boucle


while True:

    camera.start_preview()  #Lance la caméra

    camera.capture(link) # A retester : récupération de l'image à partir du str1

    camera.stop_preview() #Arrête l'affichage de la camér

    original_image = mpimg.imread(link)
    R,G,B = original_image[:,:,0],original_image[:,:,1],original_image[:,:,2] # Passage en noir et blanc
    ima = 0.2989*R + 0.5870*G + 0.1140*B
    blured = gaussian_filter(ima, sigma=1)

    im = feature.canny(blured, sigma=1, low_threshold = 50, high_threshold = 150) # Sensibilité du filtre. A regler

    height,width,j = original_image.shape # Taille de l'image

    Cmax,k = hough_transform(im)
    
    print(Cmax[0]-Rmax, Cmax[1]-Rmax, Cmax[2])
    
    # affichage(Cmax)


end_time = time.time()
time_taken = end_time - start_time
print ('Time taken for execution',time_taken)
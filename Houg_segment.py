pi = 3.1415
import math as m
import socket
from time import sleep
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
import skimage.io
from skimage.color import rgb2gray
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
from picamera import PiCamera


def hough_droite(img,pasTheta, nbrMax):
    width, height, p = img.shape # En vrai l'inverse mais pur notre axe
    print(width)
    Rmax = width+height
    tab = np.zeros((Rmax,360))
    for i in range(width):
        for j in range(height):
            if img[i][j][0] >= 1:
                for k in range(0,360,pasTheta):
                    if int(i*m.cos(m.radians(k)) + j*m.sin(m.radians(k))) >= 0:
                        tab[int(i*m.cos(m.radians(k)) + j*m.sin(m.radians(k))),k] += 1
    nb = 0
    for i in range(Rmax):
        for j in range(360):
            if tab[i,j] >= 20:
                nb += 1
    print("Nombre", nb)
            
    print(np.amax(tab))
    #plt.imshow(tab)
    #plt.show()
    ldroite = []
    while (len(ldroite) != 2):
        maxI,maxJ,max = 0,0,0
        for i in range(Rmax):
            for j in range(0,360,pasTheta):
                if tab[i][j] > max:
                    # print(i,j,tab[i][j])
                    maxI = i
                    maxJ = j
                    max = tab[i][j]
        print(max)
        tab[maxI,maxJ] = 0
        a = -1*m.tan(m.radians(maxJ))
        b = maxI*m.cos(m.radians(maxJ))-a*maxI*m.sin(m.radians(maxJ))
        if (len(ldroite)==0):
            ldroite.append([a,b]) # Ajouter dépassement
        else:
            if abs(ldroite[0][0]*a+1)<=0.2:
                ldroite.append([a,b])
        print(ldroite)
        
    print(ldroite)
    lcoodroite = []

    for i in ldroite:
        a = i[0]
        b = i[1]
        for x in range(1,height-1):
            if a*x+b>=1 and a*x+b+1<width:
                img[int(a*x+b),x,0] = 1
                img[int(a*x+b)-1,x,0] = 1
                img[int(a*x+b)+1,x,0] = 1
                img[int(a*x+b),x,0] = 1
                img[int(a*x+b)-1,x+1,0] = 1
                img[int(a*x+b)+1,x+1,0] = 1
                img[int(a*x+b),x+1,0] = 1
                img[int(a*x+b)-1,x-1,0] = 1
                img[int(a*x+b)+1,x-1,0] = 1
        print(0,int(b))
    plt.imshow(img)
    plt.show()
    return ldroite


link = '/home/pi/Desktop/image0.jpg'

camera = PiCamera() #on définit camera comme la caméra de la raspberry
camera.resolution = (1365,768) #définir la résolution on va devoir rester en 16/9


camera.start_preview()  #Lance la caméra
camera.capture(link) # A retester : récupération de l'image à partir du str1

camera.stop_preview()

#link = "C:/Users/Tanguy travail/Documents/testLignes.png"
original_image = mpimg.imread(link)
R,G,B = original_image[:,:,0],original_image[:,:,1],original_image[:,:,2] # Passage en noir et blanc

ima = 0.2989*R + 0.5870*G + 0.1140*B
blured = gaussian_filter(ima, sigma=1)


im = feature.canny(blured, sigma=1, low_threshold = 50, high_threshold = 150) # Sensibilité du filtre. A regler

img = np.zeros((764,1361,3)) # On enlève les bords
img[:,:,0] = im[2:766,2:1363]
img[:,:,1] = im[2:766,2:1363]
img[:,:,2] = im[2:766,2:1363]

#img = original_image
# img = mpimg.imread(link)
l_droite = hough_droite(img,1,2)

def droite(l_droite,im_H): # Détermine quelles droites se coupent
    x = (l_droite[0][1] - l_droite[1][1])/(l_droite[1][0] - l_droite[0][0])
    y = l_droite[0][0]*x + l_droite[0][1]
    coin = [int(x),int(y)] # La petite droite (haut) arrive en premier (axe y)
    return coin

def changementBase(C ,coin, l_droite):
    C[0] -= coin[1] # Car on inverse x et y
    C[1] -= coin[0]
    theta = m.atan(l_droite[0][0]) # On met -1 car erreur d'axe et encore -1 car angle négatif
    x = C[0]*m.cos(theta) - C[1]*m.sin(theta)
    y = C[0]*m.sin(theta) + C[1]*m.cos(theta)
    C[0] = x
    C[1] = y
    return C


coin = droite(l_droite,img) # On a les quatres coins de l'image
print(coin)
C = [550,650]

print(changementBase(C, coin, l_droite))

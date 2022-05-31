import numpy as np
import time
import math
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
import skimage.io
from skimage.color import rgb2gray
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter




## Fonction Gaussienne :
def H(i,j,k,sigma):
    A = 1/(2*np.pi*sigma**2)
    exp = np.exp(-((i-(k+1))**2+(j-(k+1))**2)/(2*sigma**2))
    H = A*exp
    return H


## Gradient d'intensité


theta_rad = theta/np.pi


q = 255
r = 255


# On va parcourir les dimensions
def grad_intensity(img,width,height):
    A = [width, height]
    Z = np.zeros(A)
    for i in range(0,width-1):
        for j in range(0,height-1):
            q = 255
            r = 255
            if (theta[i,j] <= 22.5 or theta[i,j] >=157.5):  # On fixe un angle de zéro degrés
                q = img[i, j+1]
                r = img[i,j-1]
            #quand on a une ligne on a ca


            if (theta[i,j] >= 22.5 and  theta > 67.5): # On fixe un angle de 45 degrés
                q = img[i+1,j-1]
                r = img[i-1,j+1]




            if (theta[i,j] <= 67.5 or theta[i,j] < 112.5):  # # On fixe un angle de 90 degrés
                q = img[i+1,j]
                r = img[i-1,j]


            if (theta[i,j] <= 67.5 or theta[i,j] < 112.5):   # On fixe un angle de 135 degrés
                q = img[i-1,j+1]
                r = img[i-1,j-1]


            if (img[i,j] >=  q and (img[i,j] >= r )): # on vérifie
                Z[i,j] = img[i,j]
            else:
                Z[i,j] = 0

    return Z






def double_threshold(s1,s2,Z,width,height): #on testera avec 0.04 et 0.8
    for i in range(0,width):
        for j in range(0,height):
            if (Z[i][j] < s1 ):
                Z[i][j] = 0


            if (Z[i][j] > s2 ):
                Z[i][j] = 1
            else:
                Z[i][j] = 0.5

    return Z





#
# def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
#
#     highThreshold = img.max() * highThresholdRatio; # on se ramène entre 0 et 1
#     lowThreshold = highThreshold * lowThresholdRatio;
#
#     M, N = img.shape
#     res = np.zeros((M,N))
#
#     weak = 25
#     strong = 255
#
#     strong_i, strong_j = np.where(img >= highThreshold)
#     zeros_i, zeros_j = np.where(img < lowThreshold)
#
#     weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
#
#     res[strong_i, strong_j] = strong
#     res[weak_i, weak_j] = weak
#
#     return (res, weak, strong)


# Partie test
link = 'C:/Users/ayoub/Desktop/Projet Electronique/CannyFilter/une-ping-pong.jpeg'
img = mpimg.imread(link) #On récupère l'image

[M,N,c] = np.shape(img) #Récup les dimensions


print(M)
print(N)
img1 = np.zeros((M,N))

#
# for i in range(M):
#     for j in  range(N):
#         img1[i][j] = (img[i][j][0]*0.2989 + img[i][j][1]*0.5870 + img[i][j][2]*0.1140 )
#TRop long

rgb_weights = [ 0.2989 , 0.5870, 0.1140]
img_gray = np.dot(img[...,:3],rgb_weights)


# plt.imshow(img_gray, cmap='gray') #Image  filtrée
# plt.title('Test image1 normalement en noir et blanc')
# plt.show()


H = np.array([[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]) #Filtre gaussien
print(H)


Iint = scipy.ndimage.convolve(img_gray, H, mode='constant', cval=0.0) #Image filtrée


plt.imshow(img_gray, cmap='gray') #Image  filtrée
plt.title('Test image1 normalement fitlrée noir et blanc')
plt.show()


Ix = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #Gradient selon x
Iy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) #Gradient selon y


Gradx = scipy.ndimage.convolve(img_gray, Ix, mode='constant', cval=0.0)
Grady = scipy.ndimage.convolve(img_gray, Iy, mode='constant', cval=0.0)


G = np.hypot(Gradx,Grady)

G = G/ G.max()  * 255 #on réajuste ses valeurs


plt.imshow(G, cmap='gray') #Image  filtrée
plt.title('Test image1 normalement fitlrée noir et blanc')
plt.show()

theta = np.arctan2(Grady,Gradx)  #on récup l'angle pour la transfo





A = grad_intensity(G,M,N)

plt.imshow(A, cmap='gray') #Image  filtrée
plt.title('Test image1 Gradient d intensité ')
plt.show()


# B = threshold(A,0.05,0.09)


# plt.imshow(B, cmap='gray') #Image  filtrée
# plt.title('Test image1 seuillage')
# plt.show()



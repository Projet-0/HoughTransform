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





q = 255
r = 255


# On va parcourir les dimensions
def grad_intensity(img,width,height, angle):
    A = [width, height]
    Z = np.zeros(A)
    atheta = angle
    for i in range(0,width-1):
        for j in range(0,height-1):
            a = atheta[i,j]
            q = 255
            r = 255
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            #angle 45
            elif (22.5 <= a < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            #angle 90
            elif (67.5 <= a < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            #angle 135
            elif (112.5 <= a  < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i,j] >= q) and (img[i,j] >= r):
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

# Faire un atableau avec 3 éléments, les coordonnées et la valeur on va parcourir ce qui est non nul

def threshold(img, sinf=0.05, ssup=0.09):

    Sinf = 255*sinf
    Ssup = 255*ssup



    [M,N] = np.shape(img)

    for i in range(M):
        for j in range(N):
            a = img[i][j]

            if (img[i][j] < Sinf):
                img[i][j] = 0
            if (img[i][j] > Ssup):
                img[i][j] = 255

            else:
                img[i][j] == 127

    return img


def hysteresis(img, weak, strong=255):
    M, N = np.shape(img)
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):

                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                else:
                        img[i, j] = 0
    return img





start_time = time.time()
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

# 0.06 s



# plt.imshow(img_gray, cmap='gray') #Image  filtrée
# plt.title('Test image1 normalement en noir et blanc')
# plt.show()


H = np.array([[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]) #Filtre gaussien
print(H)


Iint = scipy.ndimage.convolve(img_gray, H, mode='constant', cval=0.0) #Image filtrée


# plt.imshow(img_gray, cmap='gray') #Image  filtrée
# plt.title('Test image1 normalement fitlrée noir et blanc')
# plt.show()


Ix = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #Gradient selon x
Iy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) #Gradient selon y


Gradx = scipy.ndimage.convolve(img_gray, Ix, mode='constant', cval=0.0)
Grady = scipy.ndimage.convolve(img_gray, Iy, mode='constant', cval=0.0)



G = np.hypot(Gradx,Grady)

G = G/ G.max()  * 255 #on réajuste ses valeurs

#0.1 s



# plt.imshow(G, cmap='gray') #Image  filtrée
# plt.title('Test image1 normalement fitlrée noir et blanc')
# plt.show()

theta = np.arctan2(Grady,Gradx)  #on récup l'angle pour la transfo


angle = theta # on récupère les angles

# theta_rad = theta/np.pi





A = grad_intensity(G,M,N,angle)

#2.3


# plt.imshow(A, cmap='gray') #Image  filtrée
# plt.title('Test image1 Gradient d intensité ')
# plt.show()


B = threshold(A,0.04,0.09)



end_time = time.time()
time_taken = end_time - start_time
print ('Time taken for execution',time_taken)



# plt.imshow(B, cmap='gray') #Image  filtrée
# plt.title('Test image1 seuillage')
# plt.show()


C = hysteresis(B,20,255)

plt.imshow(C, cmap='gray') #Image  filtrée
plt.title('Test après hysterisis')
plt.show()


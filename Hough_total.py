import socket
from time import sleep
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



CAMERA = False # Choisir si l'on veut récup les images du pi ou non





start_time = time.time()

k = 0

# list1 = ['/home/pi/Desktop/image',k, '.jpg'] # Changement de type on le passe en liste pour implémenter le numéro de l'image
# str1 = ''.join(list1)  #On repasse en format string pour stocker l'image dans la camera
# print(str1)



radi = 50
pi = 3.14159

Rmin = 10
Rmax = 150

pasR = 1
pasXY = 1

az = np.full((5,5),255) # Tableau blue de 5*5

def setup_tcp(IP,port):
    adresse = ''
    serveur = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    serveur.bind((adresse, port)) #création du serveur
    serveur.listen(1) #écoute un seul client
    hostname = socket.gethostname() #nom de la machine
# IPAddr = socket.gethostbyname(hostname) #adresse IP de la machine
    print('Adresse IP du serveur: '+ IP)
    print ('Le serveur écoute sur le port:'+ str(port) )
    return serveur

donnees ='' # Donnees transmise

def envoyer(a,b,r,width,heigth):
    a = int(1000*(a/width))/1000
    b = int(1000*(b/heigth))/1000
    r = int(1000*(r/width))/1000
    donnees = str(a) + ',' + str(b) + ',' + str(r)
    client.send(donnees.encode('utf-8 '))


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

#fonction pour réduire la résolution (travail par zone) qui prendra en argument le seuil, l'"image et le pas


def resize_image(image,seuil,pas, longueur, largeur, saut) :
    new_image = np.zeros((int(longueur/pas)+1,int(largeur/pas)+1)) #matrice finale, Il faut aussi traiter les bords

    for k in range(0,longueur,pas): #on étudie toute l'image avec un
        for i in range(0,largeur,pas):
            stock = 0 #nombre de pixels blanc présent par zone
            for x in range(0,pas,saut):     #On va parcourir la zone du pas ex : 2*2)
                for y in range(0,pas,saut):
                    if (((k+x) < longueur) and ((i+y) < largeur)): # on teste le dépassement
                        if image[k+x][i+y] != 0 : # on vérifie que le pixel soit blanc
                            stock += 1 # on implémente le compteur de blanc

            if stock >= seuil: # on vérifie avec le seuil de pixel blanc
                new_image[int(k/pas)][int(i/pas)] = 1 #on le passe à l'état blanc
    print('Fin')
    print((int(longueur/pas)+1,int(largeur/pas)+1))
    return new_image


# Début

serveur = setup_tcp('192.168.139.234',2200)


while not(donnees=='end\r'):
    client, adresseClient = serveur.accept() #attente d'un client
    print ('Connexion de', adresseClient)

    while not(donnees=='stop\r' or donnees =='end\r'):

        link = '/home/pi/Desktop/image0.jpg'


        from picamera import PiCamera

        if CAMERA:
            camera = PiCamera() #on définit camera comme la caméra de la raspberry
            camera.resolution = (1024,768) #définir la résolution on va devoir rester en 16/9


            camera.start_preview()  #Lance la caméra
            camera.capture(link) # A retester : récupération de l'image à partir du str1

            camera.stop_preview() #Arrête l'affichage de la camér

        else:
            link = '/home/pi/Desktop/image0.jpg'
            link = 'C:/Users/ayoub/Desktop/Projet Electronique/APP3/HoughTransform-main/une-ping-pong.jpeg'


        original_image = mpimg.imread(link)
        R,G,B = original_image[:,:,0],original_image[:,:,1],original_image[:,:,2] # Passage en noir et blanc
        ima = 0.2989*R + 0.5870*G + 0.1140*B
        blured = gaussian_filter(ima, sigma=1)

        im = feature.canny(blured, sigma=1, low_threshold = 50, high_threshold = 150) # Sensibilité du filtre. A regler
        height,width,j = original_image.shape
        im = resize_image(im,2,4, height,width,2)#Redimensionnement nouvelle version saut de 2
        height,width = im.shape # Taille de l'image


    #On réajuste les dimensions de l'image, il faut remultiplier par le pas MAIS il faut faire attention au Rmax
        resize_pas = 4 # C'est le pas du resize
        Cmax[0] = resize_pas*(Cmax[0] - Rmax) + Rmax
        Cmax[1] = resize_pas*(Cmax[1] - Rmax) + Rmax
        Cmax[2] = resize_pas*Cmax[2]




        Cmax,k = hough_transform(im)

        print(Cmax[0]-Rmax, Cmax[1]-Rmax, Cmax[2])
        a = Cmax[0]-Rmax
        b = Cmax[1]-Rmax
        r = Cmax[2]

        height,width,j = original_image.shape # On reprend les tailles originalles
        envoyer(a,b,r,width,heigth)

        # affichage(Cmax)

    print('Fermeture de la connexion avec le client')
    client. close()
print ('Arret du serveur')
serveur.close()

end_time = time.time()
time_taken = end_time - start_time
print ('Time taken for execution',time_taken)

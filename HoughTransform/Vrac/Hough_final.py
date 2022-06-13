import socket
from time import sleep
import numpy as np
import time
import math as m
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
import skimage.io
from skimage.color import rgb2gray
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import socket
from picamera import PiCamera

## Paramètres

# Resize
faitResize = True # Fait un resize
seuilResize = 1 # Le nombre de points nécessaire par zones de l'image
pasResize = 2 # La taille des zones
sautResize = 1 # Le pas de parcourt des zones

# Hough circulaire
radi = 50 # Le nombre cercle testé à un rayon donné
Rmin = 30 //pasResize # Rayon minimum du cercle
Rmax = 60 //pasResize # Rayon maximum du cercle
pasR = 1 # Pas du rayon du test du cercle
pasXY = 1 # Pas des points gardé pour la transformé (A laisser à 1, utiliser plutôt le resize)

# Hough droite
pasTheta = 1 # Pas de l'angle en ° pour la transformé en droite
nbrDroite = 2 # Le nombre de droite à detecter (laisser à 2)

#TCP
ip = "192.168.226.178"
port = 2200

## Fin paramètres

# Constantes
pi = 3.14159
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
    r = int(1000*(r/width))/1000 # R dépend de l'axe X
    donnees = str(a) + ',' + str(b) + ',' + str(r) # Sur l'appli x et y sont inverser
    client.send(donnees.encode('utf-8 '))
    sleep(1)
    donnees = client.recv(1024)
    donnees = donnees.decode('utf-8')
    return donnees


def point(arr,radi,x,y,Cmax): # Tranformation d'un point, arr = Accumulation; radi = Nombre d'angle; x,y = Coordonées
    for r in range(Rmin,Rmax,pasR): # Pas
        for i in range(radi):
            teta = 2*pi*i/radi
            a = int(Rmax + x + r*m.cos(teta))
            b = int(Rmax + y + r*m.sin(teta))
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

def affichage(Cmax,original_image, blured, im):
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
        original_image[int(Cmax[0]-Rmax+Cmax[2]*m.cos(teta))][int(Cmax[1]-Rmax+Cmax[2]*m.sin(teta))][0] = 255
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
    print('Fin resize')
    #print((int(longueur/pas)+1,int(largeur/pas)+1))
    return new_image

def hough_droite(img,pasTheta, nbrMax, original_image):
    width, height = img.shape # En vrai l'inverse mais pur notre axe
    #print(width)
    rmax = width+height
    tab = np.zeros((rmax,360))
    for i in range(width):
        for j in range(height):
            if img[i][j] >= 1:
                for k in range(0,360,pasTheta):
                    if int(i*m.cos(m.radians(k)) + j*m.sin(m.radians(k))) >= 0:
                        tab[int(i*m.cos(m.radians(k)) + j*m.sin(m.radians(k))),k] += 1
    nb = 0
    for i in range(rmax):
        for j in range(360):
            if tab[i,j] >= 20:
                nb += 1
    #print("Nombre", nb)

    #print(np.amax(tab))
    #plt.imshow(tab)
    #plt.show()
    ldroite = []
    while (len(ldroite) != 2):
        maxI,maxJ,max = 0,0,0
        for i in range(rmax):
            for j in range(0,360,pasTheta):
                if tab[i][j] > max:
                    # print(i,j,tab[i][j])
                    maxI = i
                    maxJ = j
                    max = tab[i][j]
        #print(max)
        tab[maxI,maxJ] = 0
        a = -1*m.tan(m.radians(maxJ))
        b = maxI*m.cos(m.radians(maxJ))-a*maxI*m.sin(m.radians(maxJ))
        if (len(ldroite)==0):
            ldroite.append([a,b]) # Ajouter dépassement
        else:
            if abs(ldroite[0][0]*a+1)<=0.2:
                ldroite.append([a,b])
        #print(ldroite)

    #print(ldroite)
    lcoodroite = []

    for i in ldroite:
        a = i[0]
        b = i[1]
        for x in range(1,height-1):
            if a*x+b>=1 and a*x+b+1<width:
                original_image[int(a*x+b),x,0] = 1
                original_image[int(a*x+b)-1,x,0] = 1
                original_image[int(a*x+b)+1,x,0] = 1
                original_image[int(a*x+b),x,0] = 1
                original_image[int(a*x+b)-1,x+1,0] = 1
                original_image[int(a*x+b)+1,x+1,0] = 1
                original_image[int(a*x+b),x+1,0] = 1
                original_image[int(a*x+b)-1,x-1,0] = 1
                original_image[int(a*x+b)+1,x-1,0] = 1
        #print(0,int(b))
    #plt.imshow(img)
    #plt.show()
    return ldroite

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

## Début du code

hostname = socket.gethostname()    

serveur = setup_tcp(ip,port)

while not(donnees=='end\r'):
    donnees = ''
    client, adresseClient = serveur.accept() #attente d'un client
    print ('Connexion de', adresseClient)

    link = '/home/pi/Desktop/image0.jpg'

    camera = PiCamera() #on définit camera comme la caméra de la raspberry
    camera.resolution = (1024,768) #définir la résolution on va devoir rester en 16/9

    while not(donnees=='stop\r' or donnees =='end\r'):
    
        start_time = time.time()

        print("Début")
        
        camera.start_preview()  #Lance la caméra
        camera.capture(link) # A retester : récupération de l'image à partir du str1

        camera.stop_preview() #Arrête l'affichage de la caméra

        original_image = mpimg.imread(link)
        R,G,B = original_image[:,:,0],original_image[:,:,1],original_image[:,:,2] # Passage en noir et blanc
        ima = 0.2989*R + 0.5870*G + 0.1140*B
        blured = gaussian_filter(ima, sigma=1)
        im = feature.canny(blured, sigma=1, low_threshold = 50, high_threshold = 150) # Sensibilité du filtre. A regler
        height,width = im.shape
        
        print("Début resize")
        
        if faitResize:
            im = resize_image(im,seuilResize,pasResize, height,width,sautResize)#Redimensionnement nouvelle version saut de 2
        
        height,width = im.shape # Taille de l'image

        xim, yim = im.shape
        im = im[2:xim-2,2:yim-2] # On évité les artefacts sur les bords
        #plt.imshow(im)
        #plt.show()

    #On réajuste les dimensions de l'image, il faut remultiplier par le pas MAIS il faut faire attention au Rmax
        #resize_pas = 4 # C'est le pas du resize
        #Cmax[0] = resize_pas*(Cmax[0] - Rmax) + Rmax
        #Cmax[1] = resize_pas*(Cmax[1] - Rmax) + Rmax
        #Cmax[2] = resize_pas*Cmax[2]



        print("Début hough cercle")
        Cmax,k = hough_transform(im)
        print("Ok hough cercle")

        print("Début hough droite")
        l_droite = hough_droite(im,pasTheta,nbrDroite,original_image)
        print("Ok hough droite")

        #affichage(Cmax,original_image, blured, im)

        coin = droite(l_droite,im) # On a le coin s de l'image
        print("Ok coin")

        C = [Cmax[0]-Rmax,Cmax[1]-Rmax] # On supprime Rmax avant
        C = changementBase(C, coin, l_droite)

        #print(C[0], C[1], Cmax[2])
        a = abs(C[0])
        b = abs(C[1])
        r = Cmax[2]


        height,width = im.shape # On reprend les tailles originalles
        donnees = envoyer(a,b,r,width,height)
        
        end_time = time.time()
        
        print("Temps :")
        print(end_time - start_time)
        # affichage(Cmax)

    print('Fermeture de la connexion avec le client')
    client. close()
print ('Arret du serveur')
serveur.close()




##
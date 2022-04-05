

import numpy as np

# Commande linux pour installer la PiCamera :
# sudo apt-get install python3-picamera


## Récupération des images de la caméra

# from picamera import PiCamera
# from time import sleep



link = '/home/pi/Desktop/image0.jpg'

k = 3

# link[22] = k
int(k)
k = k+1

print(k)



k  = str(k)  # on transforme k en string

for i in range(k) :
    list1 = ['/home/pi/Desktop/image',k, '.jpg'] # Changement de type on le passe en liste pour implémenter le numéro de l'image
    str1 = ''.join(list1)  #On repasse en format string pour stocker l'image dans la camera
    print(str1)
    int(k)
    k = k+1
    k = str(k)


    list1 = ['/home/pi/Desktop/image',k, '.jpg'] # Changement de type on le passe en liste pour implémenter le numéro de l'image
    str1 = ''.join(list1)  #On repasse en format string pour stocker l'image dans la camera
    print(str1)



# camera = PiCamera() #on définit camera comme la caméra de la raspberry
# camera.resolution = (1024,768) #définir la résolution on va devoir rester en 16/9
# camera.start_preview()  #Lance la caméra
#
#
# camera.capture(str1) # A retester : récupération de l'image à partir du str1
#
#
# camera.stop_preview() #Arrête l'affichage de la caméra
import socket
from time import sleep
import random

ADRESSE = ''
PORT = 2200 #définition du port
serveur = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
serveur.bind((ADRESSE, PORT)) #création du serveur
serveur.listen(1) #écoute un seul client
donnees =''
hostname = socket.gethostname() #nom de la machine
# IPAddr = socket.gethostbyname(hostname) #adresse IP de la machine
IPAddr = '192.168.0.17'
print('Adresse IP du serveur: '+IPAddr)
print ('Le serveur écoute sur le port:'+str (PORT) )

while not(donnees=='end\r'):
    client, adresseClient = serveur.accept() #attente d'un client
    print ('Connexion de', adresseClient)
    while not(donnees=='stop\r' or donnees =='end\r'):
        a = int(100*random.random())/100
        b = int(100*random.random())/100
        c = int(100*random.random())/100
        donnees = str(a) + ',' + str(b) + ',' + str(c/4)
        client.send(donnees.encode('utf-8 '))
        sleep(1)
    print('Fermeture de la connexion avec le client')
    client. close()
print ('Arret du serveur')
serveur.close()
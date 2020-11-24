import sys, os
import cv2
import face_recognition
from pyvis.network import Network
import pickle
import networkx as nx
from face_detection import nameTransform,drawLabels
import numpy as np


#INPUT:
#known_face_names : faces existing
#face_distances : distances between faces from trombinoscope and faces detected
#
#OUTPUT
#nom : a name (or Unknown) in the trombinoscope manually chosen for the shown face
def entrerNom(known_face_names,face_distances):
    while 1:
        min_list = np.array(sorted(zip(range(len(face_distances)), face_distances), key=lambda t: t[1])[:4])

        print(min_list)

        for jfloat in np.flip(min_list[:,0])    :
            j = int(jfloat)
            print(known_face_names[j])
            print("Probabilité : " + str(1-face_distances[j]))
            print("")

        nom = input("Entrez le nom de la personne (avec un . à la place de l'espace)\n")
        if nom == "Unknown":
            cv2.destroyWindow("visage")
            break
        nom = nom + ".jpg"
        if nom in known_face_names:
            cv2.destroyWindow("visage")
            break
        else:
            print("Wrong name, try again\n")
    return nom

#INPUT:
#None
#
#OUTPUT
#Make the user correct the different labellization of people, correct the saves, the graph and the pictures
def nameCorrection():
    with open('saves\\net', 'rb') as entree:
        net = pickle.load(entree)
    for image in os.listdir('photo_AP'):
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_distances', 'rb') as entree:
            face_distances = pickle.load(entree)
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_names', 'rb') as entree:
            face_names = pickle.load(entree)
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_locations', 'rb') as entree:
            face_locations= pickle.load(entree)
        with open('saves\\known_face_names.txt', 'rb') as entree:
            known_face_names = pickle.load(entree)


        img = face_recognition.load_image_file('photo_AP' + '\\' + image)
        font = cv2.FONT_HERSHEY_DUPLEX

        # Inverse les Bleus et Rouges pour passer du RGB au BGR (pour fonctionner avec cv2)
        for i in range(len(img)):
            for j in range(len(img[i])):
                img[i][j][0], img[i][j][2] = img[i][j][2], img[i][j][0]

        for i in range(len(face_locations)):
            top, right, bottom, left = face_locations[i]
            name = face_names[i]
            nom = name
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, nameTransform(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


            print("Le nom est-il correct? (n for non)")
            cv2.namedWindow("visage", cv2.WINDOW_NORMAL)
            cv2.imshow("visage", img)

            key = cv2.waitKey(0)
            #Si l'identification est mauvaise
            if key == 110:
                #Entrée du nouveau nom
                nom = entrerNom(known_face_names,face_distances[i])
                #Supression de l'edge si il existe

                if name != "Unknown" and name != image:
                    net.remove_edge(nameTransform(name), nameTransform(image))
                if nom != "Unknown" and nom != image:
                    net.add_edge(nameTransform(nom), nameTransform(image), weight=1.)

            face_names[i] = nom
            name = nom


            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, nameTransform(name), (left + 6, bottom - 6), font, 1.0, (0,0,0), 1)

        drawLabels(img, face_locations, face_names, 'AP_labelled' + '\\' + "labelled_"+ image)
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_names', 'wb') as output:
            pickle.dump(face_names, output, pickle.HIGHEST_PROTOCOL)

    with open('saves\\net', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)

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
def entrerNom(known_face_names,face_distances,img_empty):
    images_faces_possible = []
    while 1:
        min_list = np.array(sorted(zip(range(len(face_distances)), face_distances), key=lambda t: t[1])[:4])

        print(min_list)

        for jfloat in np.flip(min_list[:,0]):
            j = int(jfloat)
            src = face_recognition.load_image_file('trombi' + '\\' + known_face_names[j])
            # # Inverse les Bleus et Rouges pour passer du RGB au BGR (pour fonctionner avec cv2)
            # for i in range(len(src)):
            #     for j in range(len(src[i])):
            #         src[i][j][0], src[i][j][2] = src[i][j][2], src[i][j][0]
            cv2.namedWindow("visage possibles", cv2.WINDOW_NORMAL)
            cv2.imshow("visage possibles", src)
            # calculate the 50 percent of original dimensions
            width = int(src.shape[1])
            height = int(src.shape[0])
            # dsize
            dsize = (int(width/height * 150), 150)
            # resize image
            resized = cv2.resize(src, dsize)

            images_faces_possible.append(resized)
            print(known_face_names[j])
            print("Probabilité : " + str(1-face_distances[j]))
            print("")


        image_cat = cv2.hconcat(images_faces_possible)

        cv2.namedWindow("visages possibles", cv2.WINDOW_NORMAL)
        cv2.imshow("visages possibles", image_cat)


        nom = input("Entrez le nom de la personne (avec un . à la place de l'espace)\n")
        if nom == "Unknown":
            cv2.destroyWindow("visage")
            cv2.destroyWindow("visage possibles")
            break
        nom = nom + ".jpg"
        if nom in known_face_names:
            cv2.destroyWindow("visage")
            cv2.destroyWindow("visage possibles")
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
        img_empty = img
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
            img_test = img_empty[top:bottom,left:right]
            print("\n")
            print(nom)
            print("Le nom est-il correct? (n for non)")


            cv2.namedWindow("visage", cv2.WINDOW_NORMAL)
            cv2.imshow("visage", img_test)

            key = cv2.waitKey(0)
            #Si l'identification est mauvaise
            if key == 110:
                #Entrée du nouveau nom
                nom = entrerNom(known_face_names,face_distances[i],img_empty)
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

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
    min_list = np.array(sorted(zip(range(len(face_distances)), face_distances), key=lambda t: t[1])[:4])
    for jfloat in np.flip(min_list[:, 0]):
        j = int(jfloat)
        src = face_recognition.load_image_file('trombi' + '\\' + known_face_names[j])
        # # Inverse les Bleus et Rouges pour passer du RGB au BGR (pour fonctionner avec cv2)


        # for i in range(len(src)):
        #     for k in range(len(src[i])):
        #         src[i][k][0], src[i][k][2] = src[i][k][2], src[i][k][0]
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        width = int(src.shape[1])
        height = int(src.shape[0])
        # dsize
        dsize = (int(width / height * 300), 300)
        #resize image
        resized = cv2.resize(src, dsize)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(resized, str(j), (20,100), font, 1, (0,255, 0), 1)
        images_faces_possible.append(resized)
        print("Image faite")
    print("Visages enregistrés")

    ids = []
    for jfloat in np.flip(min_list[:, 0]):
        ids.append(int(jfloat))

    while 1:
        print(min_list)
        for jfloat in np.flip(min_list[:,0]):
            j = int(jfloat)

            print(known_face_names[j])
            print("Probabilité : " + str(1-face_distances[j]))
            print("")


        image_cat = cv2.hconcat(images_faces_possible)


        cv2.namedWindow("visages possibles")
        cv2.imshow("visages possibles", image_cat)

        cv2.waitKey(0)

        nom = input("Entrez le numéro de la personne (-1 Pour unknown)\n")


        if nom == "-1":
            cv2.destroyWindow("visage")
            cv2.destroyWindow("visage possibles")
            nom = "Unknown"
            break
        if not nom.isnumeric():
            nom = -2
        if int(nom) in ids:
            nom = known_face_names[int(nom)]
            cv2.destroyWindow("visage")
            cv2.destroyWindow("visage possibles")
            break
        else:
            print("Le nom n'est pas reconnu, recommencez\n")
    return nom

#INPUT:
#None
#
#OUTPUT
#Make the user correct the different labellization of people, correct the saves, the graph and the pictures
def nameCorrection(ii):
    with open('saves\\net', 'rb') as entree:
        net = pickle.load(entree)
    print("go")
    for dur,image in enumerate(os.listdir('photo_AP')[ii::]):
        print(str(ii+dur)+str(" ème element"))
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_distances', 'rb') as entree:
            face_distances = pickle.load(entree)
        print("1")
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_names', 'rb') as entree:
            face_names = pickle.load(entree)
        print("2")
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_locations', 'rb') as entree:
            face_locations= pickle.load(entree)
        print("3")
        with open('saves\\known_face_names.txt', 'rb') as entree:
            known_face_names = pickle.load(entree)
        print("4")


        img = face_recognition.load_image_file('photo_AP' + '\\' + image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_empty = img
        font = cv2.FONT_HERSHEY_DUPLEX

        # Inverse les Bleus et Rouges pour passer du RGB au BGR (pour fonctionner avec cv2)
        # for i in range(len(img)):
        #     for j in range(len(img[i])):
        #         img[i][j][0], img[i][j][2] = img[i][j][2], img[i][j][0]

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

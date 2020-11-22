# coding:latin-1
import sys, os
import cv2
import face_recognition
from pyvis.network import Network
import pickle
import networkx as nx

import numpy as np


def nameTransform(name):
    rep = name[:-4]
    rep = rep.replace(".","_")
    return rep


def trombi():

    net = Network("1000px", "1000px")
    known_face_encodings = []
    known_face_names = []

    for element in os.listdir('trombi'):
        nom = nameTransform(element)
        known_face_names.append(element)
        known_image = face_recognition.load_image_file('trombi' + '\\' + element)
        known_image_encoded = face_recognition.face_encodings(known_image)[0]
        known_face_encodings.append(known_image_encoded)

        # Ajoute un node pour chaque personne différente
        net.add_node(nom, size="5", title=nom)

    with open('saves\\net.trombi', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
    with open('saves\\known_face_encodings.txt', 'wb') as output:
        pickle.dump(known_face_encodings, output, pickle.HIGHEST_PROTOCOL)
    with open('saves\\known_face_names.txt', 'wb') as output:
        pickle.dump(known_face_names, output, pickle.HIGHEST_PROTOCOL)

def detecte_visages():

    # Charge les embeddings des visages connus
    with open('saves\\net.trombi', 'rb') as input:
        net = pickle.load(input)
    with open('saves\\known_face_encodings.txt', 'rb') as input:
        known_face_encodings = pickle.load(input)
    with open('saves\\known_face_names.txt', 'rb') as input:
        known_face_names = pickle.load(input)

    #cherche les visages sur chaque photo AP
    for image in os.listdir('photo_AP'):
        # on charge l'image en mémoire
        img = face_recognition.load_image_file('photo_AP' + '\\' +image)
        image_out = 'AP_labelled' + '\\' + "labelled_"+ image

        # Trouve les visages sur la photo
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)


        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            # else:
            #     face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            #     best_match_index = np.argmin(face_distances)
            #     name = known_face_names[best_match_index] + '(guess)'
            face_names.append(name)


        for n1 in face_names:
            for n2 in face_names:
                if n1 != "Unknown" and n2 != "Unknown" and n1 != n2:
                    net.add_edge(nameTransform(n1),nameTransform(n2), weight=1.)

        # Inverse les Bleus et Rouges pour passer du RGB au BGR (pour fonctionner avec cv2)
        for i in range(len(img)):
            for j in range(len(img[i])):
                img[i][j][0],img[i][j][2] = img[i][j][2],img[i][j][0]


        for i in range(len(face_locations)):
            top, right, bottom, left = face_locations[i]
            name = face_names[i]

            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, nameTransform(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imwrite(image_out, img)

    # Create data of neigbours
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node["title"] += " neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
        node["value"] = 1
    net.show("graphe.html")

if __name__ == "__main__":
    # applique
    trombi()
    print("Trombinoscope saved")
    detecte_visages()
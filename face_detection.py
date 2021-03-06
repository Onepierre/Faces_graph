# coding:latin-1
import sys, os
import cv2
import face_recognition
from pyvis.network import Network
import pickle
import networkx as nx
import time
from label_correction import *





#INPUT:
#name : name of a a picture (prenom.nom.jpg)
#
#OUTPUT
#rep : adapted name (prenom_nom)
def nameTransform(name):
    if name == "Unknown":
        return name
    rep = name[:-4]
    rep = rep.replace(".","_")
    return rep






#INPUT:
#img : screen with people
#face_locations : locations of the faces
#face_names : actual names of the faces
#image_out : name of the output screen
#
#OUTPUT
#save and return the photo with laballed faces
def drawLabels(img,face_locations,face_names,image_out,save = True):
    for i in range(len(face_locations)):
        top, right, bottom, left = face_locations[i]
        name = face_names[i]
        # Draw a box around the face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, nameTransform(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    if save:
        cv2.imwrite(image_out, img)
    return img

#INPUT:
#None
#
#OUTPUT
#Create and save the laballed representations of everybody
#Create and save the graph of people
def createTrombi(count = False):
    net = nx.MultiDiGraph()
    known_face_encodings = []
    known_face_names = []
    n = len(os.listdir('trombi'))
    for i,element in enumerate(os.listdir('trombi')):
        if count:
            print(str(i) +"/" + str(n))
        nom = nameTransform(element)
        known_face_names.append(element)
        known_image = face_recognition.load_image_file('trombi' + '\\' + element)
        known_image_encoded = face_recognition.face_encodings(known_image)[0]
        known_face_encodings.append(known_image_encoded)

        # Ajoute un node pour chaque personne diff�rente
        net.add_node(nom, size="5", title=nom)

    with open('saves\\net', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
    with open('saves\\known_face_encodings.txt', 'wb') as output:
        pickle.dump(known_face_encodings, output, pickle.HIGHEST_PROTOCOL)
    with open('saves\\known_face_names.txt', 'wb') as output:
        pickle.dump(known_face_names, output, pickle.HIGHEST_PROTOCOL)


#INPUT:
#None
#
#OUTPUT
#Create and save the laballed representations of everybody on the photos containing several people
#Edit and save the graph of link between people appearing on the same photo
def detectFaces(count = False):
    t0 = time.time()
    # Charge les embeddings des visages connus
    with open('saves\\net', 'rb') as entree:
        net = pickle.load(entree)
    with open('saves\\known_face_encodings.txt', 'rb') as entree:
        known_face_encodings = pickle.load(entree)
    with open('saves\\known_face_names.txt', 'rb') as entree:
        known_face_names = pickle.load(entree)
    n = len(os.listdir('photo_AP'))
    #cherche les visages sur chaque photo AP
    for i,image in enumerate(os.listdir('photo_AP')):
        if count:
            text1= str(i) + "/" + str(n)
            t1 = time.time()
            if i>0:
                tps = time.gmtime((t1-t0))
                text2 = str(tps[3]) + " h " + str(tps[4]) + " min " \
                        + str(tps[5]) + " sec " + "effectu�es."
                tps2 = (t1-t0)*n/i
                tps = time.gmtime(tps2 - (t1 - t0))
                text3=str(tps[3]) +" h "+str(tps[4]) +" min "+str(tps[5]) +" sec " +"restantes."
                text4 = "-------------------------------------------"
                print(text1 +"\n" +text2 +"\n" +text3 +"\n" + text4, end="\r")
            else:
                print(text1, end = "\r")
        # on charge l'image en m�moire
        img = face_recognition.load_image_file('photo_AP' + '\\' +image)
        image_out = 'AP_labelled' + '\\' + "labelled_"+ image
        # Trouve les visages sur la photo
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        face_names = []
        face_distances = []

        # Cherche � identifier chaque visage
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            #matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            t_face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            min = sorted(zip(range(len(t_face_distances)), t_face_distances), key=lambda t: t[1])[0]

            name = "Unknown"

            if min[1]<0.5:
                name = known_face_names[min[0]]

            face_names.append(name)
            face_distances.append(t_face_distances)

        with open('saves\\Recognition\\' + nameTransform(image) +'face_distances', 'wb') as output:
            pickle.dump(face_distances, output, pickle.HIGHEST_PROTOCOL)
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_names', 'wb') as output:
            pickle.dump(face_names, output, pickle.HIGHEST_PROTOCOL)
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_encodings', 'wb') as output:
            pickle.dump(face_encodings, output, pickle.HIGHEST_PROTOCOL)
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_locations', 'wb') as output:
            pickle.dump(face_locations, output, pickle.HIGHEST_PROTOCOL)

        # add edges to the graph
        for n1 in face_names:
            if n1 != "Unknown" and n1 != image:
                net.add_edge(nameTransform(n1),nameTransform(image), weight=1.,color = "black")

        # Inverse les Bleus et Rouges pour passer du RGB au BGR (pour fonctionner avec cv2)
        for i in range(len(img)):
            for j in range(len(img[i])):
                img[i][j][0],img[i][j][2] = img[i][j][2],img[i][j][0]

        # draw the rectangles on the picture
        drawLabels(img, face_locations, face_names, image_out)

    with open('saves\\net', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)



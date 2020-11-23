# coding:latin-1
import sys, os
import cv2
import face_recognition
from pyvis.network import Network
import pickle
import networkx as nx



def nameTransform(name):
    if name == "Unknown":
        return name
    rep = name[:-4]
    rep = rep.replace(".","_")
    return rep

def printGraph(graph):
    net = Network("1000px", "1000px",directed=True)
    net.from_nx(graph)
    # Create data of neigbours
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node["title"] += " neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
        node["value"] = 1
    net.show("graphe.html")


def trombi():
    net = nx.DiGraph()
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

    with open('saves\\net', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
    with open('saves\\known_face_encodings.txt', 'wb') as output:
        pickle.dump(known_face_encodings, output, pickle.HIGHEST_PROTOCOL)
    with open('saves\\known_face_names.txt', 'wb') as output:
        pickle.dump(known_face_names, output, pickle.HIGHEST_PROTOCOL)

def detecte_visages():

    # Charge les embeddings des visages connus
    with open('saves\\net', 'rb') as entree:
        net = pickle.load(entree)
    with open('saves\\known_face_encodings.txt', 'rb') as entree:
        known_face_encodings = pickle.load(entree)
    with open('saves\\known_face_names.txt', 'rb') as entree:
        known_face_names = pickle.load(entree)

    #cherche les visages sur chaque photo AP
    for image in os.listdir('photo_AP'):

        # on charge l'image en mémoire
        img = face_recognition.load_image_file('photo_AP' + '\\' +image)
        image_out = 'AP_labelled' + '\\' + "labelled_"+ image
        # Trouve les visages sur la photo
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        face_names = []
        face_distances = []

        # Cherche à identifier chaque visage
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            face_names.append(name)
            face_distances.append(face_distance)

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
                net.add_edge(nameTransform(n1),nameTransform(image), weight=1.)

        # Inverse les Bleus et Rouges pour passer du RGB au BGR (pour fonctionner avec cv2)
        for i in range(len(img)):
            for j in range(len(img[i])):
                img[i][j][0],img[i][j][2] = img[i][j][2],img[i][j][0]

        # draw the rectangles on the picture
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

    with open('saves\\net', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)

    printGraph(net)

def nameCorrection():
    with open('saves\\net', 'rb') as entree:
        net = pickle.load(entree)
    printGraph(net)
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

            key = cv2.waitKey()
            print(face_names)
            if key == 110:
                while 1:
                    for j in range(len(known_face_names)):
                        print(known_face_names[j])
                        print(face_distances[i][j])
                    nom = input("Entrez le nom de la personne (avec un . à la place de l'espace)")
                    if nom == "Unknown":
                        cv2.destroyWindow("visage")
                        break
                    nom = nom + ".jpg"
                    if nom in known_face_names:
                        cv2.destroyWindow("visage")
                        break
                    else:
                        print(known_face_names)

                if name != "Unknown" and name != image:
                    net.remove_edge(nameTransform(name), nameTransform(image))
                if nom != "Unknown" and nom != image:
                    net.add_edge(nameTransform(nom), nameTransform(image), weight=1.)

            face_names[i] = nom
            name = nom


            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, nameTransform(name), (left + 6, bottom - 6), font, 1.0, (0,0,0), 1)

    with open('saves\\net', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
    with open('saves\\Recognition\\' + nameTransform(image) + 'face_names', 'wb') as output:
        pickle.dump(face_names, output, pickle.HIGHEST_PROTOCOL)
    printGraph(net)

if __name__ == "__main__":
    # applique
    # trombi()
    # print("Trombinoscope saved")
    # detecte_visages()
    # print("detect visage terminé")
    with open('saves\\net', 'rb') as entree:
        net = pickle.load(entree)
    printGraph(net)

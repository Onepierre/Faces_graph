import sys, os
import cv2
import face_recognition
from pyvis.network import Network
import pickle
import networkx as nx
import numpy as np
import pandas as pd





def nameTransform(name):
    if name == "Unknown":
        return name
    rep = name[:-4]
    rep = rep.replace(".","_")
    return rep





def rebuildGraph():
    graph = nx.MultiDiGraph()
    with open('saves\\known_face_names.txt', 'rb') as entree:
        known_face_names = pickle.load(entree)
    for nom in known_face_names:
        graph.add_node(nameTransform(nom), size=1, title=nom)

    for image in os.listdir('photo_AP'):
        with open('saves\\Recognition\\' + nameTransform(image) + 'face_names', 'rb') as entree:
            face_names = pickle.load(entree)
        for n1 in face_names:
            if n1 != "Unknown" and n1 != image:
                graph.add_edge(nameTransform(n1),nameTransform(image), weight=1.,color = "black",title = "Same AP photo")

    return graph


#INPUT:
#name : a name (prenom_nom)
#
#OUTPUT
#return the name in a more beautiful version with the caps
def prettyName(name):
    return name.replace('_',' ').replace('.',' ').title()




#INPUT:
#line : a line representing a person
#node : the node to modify
#
#OUTPUT
#return the modified node to print the binets
def addSectionNodes(graph,data):
    nodes_values = []
    for node in graph.nodes:
        nodes_values.append(data[data["nom"] == node.replace("_", ".")])
    for i,node in enumerate(nodes_values):
        for j,node2 in enumerate(nodes_values):
            if i != j and node["section"].values[0] == node2["section"].values[0]:
                graph.add_edge(node["nom"].values[0].replace(".", "_"),node2["nom"].values[0].replace(".", "_"),
                               weight=1.,color = "blue",hidden = False,physics = False, title = "Same section")
    return graph



#INPUT:
#line : a line representing a person
#selected : if the point is selected or non
#
#OUTPUT
#return the color (red or yellow) of his promotion
def colorOfPromotion(line,selected = False):
    text = line["promo"].values[0]
    promo = int(text[1:])
    if promo/2 == promo//2:
        if selected:
            return "darkred"
        return "red"
    if selected:
        return "gold"
    return "yellow"

#INPUT:
#line : a line representing a person
#node : the node to modify
#
#OUTPUT
#return the modified node to print the binets
def displayBinets(line,node):
    binets = str(line["binets"].values[0]).split(", ")
    if binets[0] == "nan":
        return node
    node["title"] += "<br>Binets : <ul>"
    for binet in binets:
        node["title"] += "<li>" + str(binet)
    node["title"] += "</ul>"
    return node




#INPUT:
#graph : the graph to print
#
#OUTPUT
#print the graph
def printGraph(graph,data):
    net = Network("500px", "500px",directed=True)
    net.from_nx(graph)
    #import data of people
    # Create data of neighbours
    # neighbor_map = net.get_adj_list()
    for node in net.nodes:
        # Get the data of the current node
        line = data[data["nom"] == node["id"].replace("_",".")]
        # Put the data
        node["label"] = prettyName(str(line["nom"].values[0]))
        node["title"] = "<b>" + prettyName(str(line["nom"].values[0])) + "</b>"
        node["title"] += "<br>Promotion : " + str(line["promo"].values[0])
        node["title"] += "<br>Section : " + str(line["section"].values[0])
        node = displayBinets(line,node)
        node["value"] = 1
        node["color"] = {"background":colorOfPromotion(line),
                         "border":"black",
                         "highlight": {"background":colorOfPromotion(line,selected = True),
                                       "border":"black"}}
        node["borderWidth"] = 3
    net.set_edge_smooth('dynamic')

    #net.show_buttons()
    net.show("graph\graphe.html")

def printGraphNoData(graph):
    net = Network("800px", "800px",directed=True)
    net.from_nx(graph)
    #import data of people
    # Create data of neighbours
    # neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node["title"] = node["id"]
        node["value"] = 1
        node["color"] = {"background":"red",
                         "border":"black",
                         "highlight": {"background":"red",
                                       "border":"black"}}
        node["borderWidth"] = 1
    #net.set_edge_smooth('dynamic')
    net.barnes_hut()
    #net.show_buttons()
    net.show("graph\graphe.html")
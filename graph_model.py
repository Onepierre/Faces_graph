import sys, os
import cv2
import face_recognition
from pyvis.network import Network
import pickle
import networkx as nx
import numpy as np
import pandas as pd


#INPUT:
#name : a name (prenom_nom)
#
#OUTPUT
#return the name in a more beautiful version with the caps
def prettyName(name):
    return name.replace('_',' ').replace('.',' ').title()


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
def printGraph(graph):
    net = Network("800px", "1400px",directed=True)
    net.from_nx(graph)
    #import data of people
    data = pd.read_csv("donnees.csv",sep=';', encoding='latin-1')
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
    for edge in net.edges:
        edge["color"] = "blue"
    net.show("graph\graphe.html")
# coding:latin-1
import sys, os
import cv2
import face_recognition
from pyvis.network import Network
import pickle
import networkx as nx

from label_correction import *
from face_detection import *

if __name__ == "__main__":
    # applique
    # trombi()
    # print("Trombinoscope saved")
    # detecte_visages()
    # print("detect visage terminé")
    with open('saves\\net', 'rb') as entree:
        net = pickle.load(entree)
    printGraph(net)
    nameCorrection()
    with open('saves\\net', 'rb') as entree:
        net = pickle.load(entree)
    printGraph(net)

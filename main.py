# coding:latin-1
import sys, os
import cv2
import face_recognition
from pyvis.network import Network
import pickle
import networkx as nx
from graph_model import *
from label_correction import *
from face_detection import *

if __name__ == "__main__":

    # createTrombi()
    # print("Trombinoscope saved")
    # detectFaces()
    # print("detect visage terminé")
    # nameCorrection()
    with open('saves\\net', 'rb') as entree:
        net = pickle.load(entree)
    printGraph(net)

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

def rename():
    for image in os.listdir('photo_AP'):
        nom2 = image.split("_")[2]+'.jpg'
        os.rename('photo_AP/' + image,'photo_AP/' + nom2)

    for image in os.listdir('trombi'):
        nom2 = image.split("_")[2] + '.jpg'
        os.rename('trombi/' +image, 'trombi/' +nom2)


if __name__ == "__main__":

    # rename()
    # createTrombi(count = True)
    # # print("Trombinoscope saved")
    #detectFaces(count = True)
    #print("detect visage terminé")
    nameCorrection(178)

    with open('saves\\net', 'rb') as entree:
        net = pickle.load(entree)
    #data = pd.read_csv("donnees.csv",sep=';', encoding='latin-1')
    #     # net = rebuildGraph()
    #     # #addSectionNodes(net,data)
    #     # printGraphNoData(net)

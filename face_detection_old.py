# coding:latin-1
import sys, os
import cv2


def detecte_visages(image, image_out, show=False):
    # on charge l'image en mémoire
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dimensions = img.shape
    print(dimensions)
    # on charge le modèle de détection des visages
    #face_model = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_model = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_model = cv2.CascadeClassifier('haarcascade_smile.xml')


    # détection du ou des visages
    faces = face_model.detectMultiScale(gray,
                                        minNeighbors = 1,
                                        minSize = (int(dimensions[0]/7),int(dimensions[1]/7)),
                                        maxSize =(int(dimensions[0]/3),int(dimensions[1]/3)))

    # on place un cadre autour des visages
    print ("nombre de visages", len(faces), "dimension de l'image", img.shape, "image", image)
    for (ex, ey, ew, eh) in faces:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255,0, 0), 2)

    # détection des yeux
    eyes = eye_model.detectMultiScale(gray,minNeighbors = 4, minSize = (30,30), maxSize =(60,60))
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # détection des sourires
    smiles = smile_model.detectMultiScale(gray, minNeighbors = 28, minSize = (100,40), maxSize =(150,60))
    for (ex, ey, ew, eh) in smiles:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    # on sauvegarde le résultat final
    cv2.imwrite(image_out, img)

    # pour voir l'image, presser ESC pour sortir
    if show:
        cv2.imshow("visage", img)
        if cv2.waitKey(5000) == 27: cv2.destroyWindow("visage")


if __name__ == "__main__":
    # applique
    for file in os.listdir("."):
        if file.startswith("visage"): continue  # déjà traité
        if os.path.splitext(file)[-1].lower() in [".jpg", ".jpeg", ".png"]:
            detecte_visages(file, "visage_" + file)
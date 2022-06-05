import os
import cv2
import sys

imagePath = sys.argv[1]

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=3,minSize=(100, 100))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    faceimage = image[y:y + h, x:x + w]
    lastimage = cv2.resize(faceimage, (256, 256))
    cv2.imwrite(str(w) + str(h) + '_crop.jpg', lastimage)

import cv2
import numpy as np
import matplotlib.pyplot as plt

#pathf = '/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
#pathf = '/home/pi/opencv/data/haarcascades/haarcascade_frontalface_alt.xml'
pathf = '/home/pi/opencv/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml'

faceCascade = cv2.CascadeClassifier(pathf)

#pathe = '/home/pi/opencv/data/haarcascades/haarcascade_eye.xml'
#eyeCascade = cv2.CascadeClassifier(pathe)

#patheg = '/home/pi/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
#eyeglassesCascade = cv2.CascadeClassifier(patheg)

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5,5))
    print('faces : ', faces)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+w), (0,255,0), 2)
    cv2.imshow('frame', frame)
    c = cv2.waitKey(1)
    if c==27: #esc key
        break
    
cv2.waitKey()
cap.release()
cv2.destroyAllWindows()
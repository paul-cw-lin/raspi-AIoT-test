import cv2
import numpy as np

cap = cv2.VideoCapture('/home/pi/opencv/opencv-master/samples/data/vtest.avi')

bg = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17,17), 0)
    
    if bg is None:
        bg = gray
        continue
    
    diff = cv2.absdiff(gray, bg)
    ret, binary = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY)
    
    erode = cv2.erode(binary, None, iterations=2)
    dilate = cv2.dilate(erode, None, iterations=2)
    
    cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        if cv2.contourArea(c) < 400:
            continue
        
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
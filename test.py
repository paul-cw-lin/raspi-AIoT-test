import time
import cv2

cap = cv2.VideoCapture(0)

print(time.time())
#print('current time : ', time.ctime())
print('Hello')
time.sleep(1.0)
print('World')

faceCascade = cv2.CascadeClassifier('/home/pi/opencv/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')
#faceCascade = cv2.CascadeClassifier('/home/pi/opencv/opencv-master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')


while cap.isOpened():
    ok, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    #print('faces : ', faces)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
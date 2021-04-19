import cv2

faceCascade = cv2.CascadeClassifier('/home/pi/opencv/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')
eyeCascade = cv2.CascadeClassifier('/home/pi/opencv/opencv-master/data/haarcascades/haarcascade_eye.xml')

o= cv2.imread('wedding.bmp')
image = o

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.1, 3)

for (x,y,w,h) in faces:
    image = cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    
    face_rect = gray[y:y+h, x:x+w]
        
    eyes = eyeCascade.detectMultiScale(face_rect, 1.01, 8)
    
    for (ex, ey, ew, eh) in eyes:
        center = (x+ey+int(ew/2.0), y+ey+int(eh/2.0))
        r = int(min(ew, eh) / 2.0)
        image = cv2.circle(image, r, (255,0,0), 2)
    
cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
cv2.imshow('video', image)

cv2.waitKey()
cv2.destroyAllWindows()
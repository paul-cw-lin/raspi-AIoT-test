import cv2

faceCascade = cv2.CascadeClassifier('/home/pi/opencv/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')

o= cv2.imread('wedding.bmp')
image = o

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.1, 3)
print('faces :\n', faces)

for (x,y,w,h) in faces:
    image = cv2.rectangle(image, (x,y), (x+w, y+h), (255,255,0), 2)
            
cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
cv2.imshow('video', image)

cv2.waitKey()
cv2.destroyAllWindows()
import cv2

eyeCascade = cv2.CascadeClassifier('/home/pi/opencv/opencv-master/data/haarcascades/haarcascade_eye.xml')

o= cv2.imread('wedding.bmp')
image = o

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eyes = eyeCascade.detectMultiScale(gray, 1.05, 8)
    
for (ex, ey, ew, eh) in eyes:
    center = (x+ey+int(ew/2.0), y+ey+int(eh/2.0))
    r = int(min(ew, eh) / 2.0)
    image = cv2.circle(image, r, (255,0,0), 2)
    
cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)
cv2.imshow('video', image)

cv2.waitKey()
cv2.destroyAllWindows()
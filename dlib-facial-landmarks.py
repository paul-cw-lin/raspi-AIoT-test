#Dlib facial landmarks
import cv2
import dlib
import numpy

PREDICTOR_PATH = '/home/pi/dlib-models/shape_predictor_5_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path = '/home/pi/opencv/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml'
cascade = cv2.CascadeClassifier(cascade_path)

def get_landmarks(img):
    rects = cascade.detectMultiScale(img, 1.3, 8)
    for (x,y,w,h) in rects:
        rect=dlib.rectangle(x,y,x+w,y+h)
        dimface = numpy.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
        img = annotate_landmarks(img, dimface)
    return img

def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    return img

o = cv2.imread('nanami.jpg')
img = o
cv2.imshow('output', get_landmarks(img))

#cv2.imwrite(‘output.jpg’, get_landmarks(im))

cv2.waitKey()
cv2.destroyAllWindows()
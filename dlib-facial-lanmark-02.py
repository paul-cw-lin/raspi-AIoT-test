import dlib
import cv2

o = cv2.imread('nanami.jpg')
img = o

predictor_path = '/home/pi/dlib-models/shape_predictor_68_face_landmarks.dat'


def renderFace (im, landmarks, color=(0,255,0), radius=3):
    for p in landmarks.parts():
        cv2.circle(im, (p.x, p.y), radius, color, -1)
        
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

dets = detector(img, 1)

for k, d in enumerate(dets):
    shape = predictor(img, d)
    renderFace(img, shape)
    
cv2.imshow('face-rendered', img)
cv2.waitKey()
cv2.destroyAllWindows()
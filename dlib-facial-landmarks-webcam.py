import dlib
import cv2

def renderFace (im, landmarks, color=(0,255,0), radius=3):
    for p in landmarks.parts():
        cv2.circle(im, (p.x, p.y), radius, color, -1)
        
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/pi/dlib-models/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    dets = detector(frame, 1)
    
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        renderFace(frame, shape)
              
    cv2.imshow('video', frame)
        
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
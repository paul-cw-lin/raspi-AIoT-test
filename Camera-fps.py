import cv2
import time

cap = cv2.VideoCapture(0)

while cap.isOpened():
    beginTime = time.time()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640,480))
    
    fps = 1/(time.time() - beginTime)
    text = 'fps:{:.1f}' .format(fps)
    cv2.putText(frame, text, (10,470), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
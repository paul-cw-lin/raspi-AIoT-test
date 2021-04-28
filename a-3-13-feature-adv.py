import cv2

orb_feature = cv2.ORB_create()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    orb_kp = orb_feature.detect(frame)
    orb_out = cv2.drawKeypoints(frame, orb_kp, None)
    cv2.imshow('frame', orb_out)
    
    c = cv2.waitKey(1)
    if c==27 or c == ord('q'): #esc key
        break

cap.release()
cv2.destroyAllWindows()
import dlib
import cv2
import time

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

while True:
    begin_time = time.time()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    face_rects = detector(frame, 0) # 偵測人臉
    
    for i, d in enumerate(face_rects): # 取出所有偵測的結果
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 以方框標示偵測的人臉
    
    fps = 1/ (time.time() - begin_time)
    text = "fps: {:.1f}" .format(fps)
    cv2.putText(frame, text, (10,20), FONT, 0.8, (255,0,0), 2)
    
    cv2.imshow('video', frame)
        
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
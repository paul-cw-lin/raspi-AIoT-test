import cv2

palmCascade = cv2.CascadeClassifier(
    '/home/pi/Documents/Xml-model/sandeep/rpalm.xml')
fistCascade = cv2.CascadeClassifier(
    '/home/pi/Documents/Xml-model/sandeep/fist.xml')


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    palm = palmCascade.detectMultiScale(gray, 1.1, 3)
    fist = fistCascade.detectMultiScale(gray, 1.1, 3)
    #print('palm : ', palm)

    for (x,y,w,h) in palm:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2)
        
    for (a,b,c,d) in fist:
        frame = cv2.rectangle(frame, (a,b), (a+c, b+d), (0,0,255), 2)
       
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
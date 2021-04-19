import cv2

ESC = 27

n = 1
index = 0
total = 100

def saveImage(face_image, index):
    filename = '/home/pi/Documents/raspi-AIoT-test/images/h0/{:03d}.jpg'.format(index)
    cv2.imwrite(filename, face_image)
    print(filename)


faceCascade = cv2.CascadeClassifier('/home/pi/opencv/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE)

while n > 0:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, 1.1, 3)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,
                              (x,y), (x+w, y+h),
                              (0,255,0), 2)
        if n % 5 == 0:
            face_img = gray[y: y+h, x: x+w]
            face_img = cv2.resize(face_img, (400,400))
            saveImage(face_img, index)
            index += 1
            if index >= total:
                print('get training data done')
                n = -1
                break
        n += 1
        
    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ESC:
        cv2.destroyAllWindows()
        break
import cv2
import time

#Tensorflow
net = cv2.dnn.readNet(
    '/home/pi/opencv/opencv-master/samples/dnn/face_detector/opencv_face_detector.pbtxt',
    '/home/pi/opencv/opencv-master/samples/dnn/face_detector/opencv_face_detector_uint8.pb')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(300,300), scale=1.0)

n = 1
index = 0
total = 100

def saveImage(face_image, index):
    filename = '/home/pi/Documents/raspi-AIoT-test/images/f0/{:03d}.jpg'.format(index)
    cv2.imwrite(filename, face_image)
    print(filename)

cap = cv2.VideoCapture(0)


while n > 0:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    classes, confs, boxes = model.detect(frame, 0.5)
    
    for (x,y,w,h) in boxes:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
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
    
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break



'''
while True:
    begin_time = time.time()
    ret, frame = cap.read()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.flip(frame, 1)
    
    classes, confs, boxes = model.detect(frame, 0.5)
    for (classid, conf, box) in zip(classes, confs, boxes):
        x,y,w,h = box
        text = '%2f' % conf
        
        if y - 20 < 0:
            y1 = y+20
        else:
            y1=y-10
            
        fps = 1/ (time.time() - begin_time)
        #text = "fps: {:.1f} {:.2f}%" .format(fps, float(conf) * 100)
        text = "fps: {:.1f}" .format(fps)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, text, (x,y-10), FONT, 1.0, (0,255,0), 2)
        
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
'''
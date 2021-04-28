import cv2
import time

'''
model_path = "/home/pi/build/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
pbtxt_path = "/home/pi/build/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.bin"

net = cv2.dnn.readNet(model_path, pbtxt_path)
'''

net = cv2.dnn.readNet(
    '/home/pi/opencv/opencv-master/samples/dnn/face_detector/opencv_face_detector.pbtxt',
    '/home/pi/opencv/opencv-master/samples/dnn/face_detector/opencv_face_detector_uint8.pb')

'''
net = cv2.dnn.readNet(
    "/home/pi/build/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
    "/home/pi/build/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.bin"
    )
'''

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

cap = cv2.VideoCapture(0)

frameID = 0

grabbed = True

start_time = time.time()

while grabbed:
    grabbed, img = cap.read()
    img = cv2.resize(img, (480,320))
    frame = img.copy()
    
    blob = cv2.dnn.blobFromImage(frame, size=(672,384), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()
    
    for detection in out.reshape(-1,7):
        confidence = float(detection[2])
        xmin = int(detection[3]*frame.shape[1])
        ymin = int(detection[4]*frame.shape[0])
        xmax = int(detection[5]*frame.shape[1])
        ymax = int(detection[6]*frame.shape[0])
        
        if confidence > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0,255,0))
    
    cv2.imshow('frame', frame)
    frameID += 1
    
    fps = frameID / (time.time()-start_time)
    print('FPS', fps)
    
    cv2.waitKey()
    cap.release()
    cv2.destroyAllWindows()
import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('/home/pi/opencv/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')

images = []
images.append(cv2.imread('/home/pi/Documents/raspi-AIoT-test/images/h0/000.jpg', cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('/home/pi/Documents/raspi-AIoT-test/images/h0/001.jpg', cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('/home/pi/Documents/raspi-AIoT-test/images/h0/002.jpg', cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('/home/pi/Documents/raspi-AIoT-test/images/h1/000.jpg', cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('/home/pi/Documents/raspi-AIoT-test/images/h1/001.jpg', cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('/home/pi/Documents/raspi-AIoT-test/images/h1/002.jpg', cv2.IMREAD_GRAYSCALE))

labels = [0,0,0,1,1,1]

names = ['Paul', 'Cloud']

print('training...')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))
recognizer.save('/home/pi/Documents/raspi-AIoT-test/model/faces-photo.data')
print('training DONE!')

predict_image = cv2.imread('/home/pi/Documents/raspi-AIoT-test/images/h1/008.jpg', cv2.IMREAD_GRAYSCALE)

faces = faceCascade.detectMultiScale(predict_image, 1.1, 3)

for (x,y,w,h) in faces:
    image = cv2.rectangle(predict_image, (x,y), (x+w, y+h), (255,255,0), 2)


label, confidence = recognizer.predict(predict_image)
print('label= ', label)
print('confidence= ', '%.1f' %confidence)

if confidence < 50:
    cv2.putText(image, names[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('image', image)

cv2.waitKey()
cv2.destroyAllWindows()

import cv2
import time

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
WIDTH = 320
HEIGHT = int(WIDTH/ratio)

cap = cv2.VideoCapture(0)
time.sleep(3)

ret, frame = cap.read()
if ret:
    cv2.imwrite('image.jpg', frame)
    print('Done')
else:
    print('Read ERROR!')
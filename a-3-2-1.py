import cv2

ESC = 27

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print('w = ', w)
print('h = ', h)

ratio = w/h 

WIDTH = 320
HEIGHT = int(WIDTH/ratio)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ESC:
        cv2.destroyAllWindows()
        break
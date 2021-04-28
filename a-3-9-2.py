import cv2

bs = cv2.bgsegm.createBackgroundSubtractorMOG()
cap = cv2.VideoCapture('/home/pi/opencv/opencv-master/samples/data/vtest.avi')
#ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

#WIDTH = 400
#HEIGHT = int(WIDTH/ratio)

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (WIDTH, HEIGHT))
    #frame = cv2.flip(frame, 1)
    
    gray = bs.apply(frame)
    ret, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    erode = cv2.erode(binary, None, iterations=2)
    dilate = cv2.dilate(erode, None, iterations=10)
    
    cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        if cv2.contourArea(c) <200:
            continue
        cv2.drawContours(frame, cnts, -1, (255,255,0), 2)
        #x,y,w,h = cv2.boundingRect(c)
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
    #frame_h = cv2.hconcat([frame, dilate])
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
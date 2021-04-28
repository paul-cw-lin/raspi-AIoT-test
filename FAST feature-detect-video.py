import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
fast.setNonmaxSuppression(True)
fast.setThreshold(40)

'''
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
'''

while cap.isOpened():
    ok, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    kp = fast.detect(frame,None)
    frame = cv2.drawKeypoints(frame, kp, None, color=(255,0,0))
    
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2
from matplotlib import pyplot as plt

o = cv2.imread('img25.jpg',0)
img = o

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
fast.setNonmaxSuppression(True)
fast.setThreshold(20)
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

cv2.imshow('img2', img2)

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

img3 = cv2.drawKeypoints(img, kp, None, color=(0,255,0))

#cv2.imshow('img3', img3)

cv2.waitKey()
cv2.destroyAllWindows()
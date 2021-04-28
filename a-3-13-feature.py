import cv2

img = cv2.imread('img25.jpg')

#sift_feature = cv2.xfeatures2d.SIFT_create()
#surf_feature = cv2.xfeatures2d.SURF_create()
orb_feature = cv2.ORB_create()

#sift_kp = sift_feature.detect(img)
#surf_kp = surf_feature(img)
orb_kp = orb_feature.detect(img)

#sift_out = cv2.drawKeypoints(img, sift_kp, None)
#surf_out = cv2.drawKeypoints(img, surf_kp, None)
orb_out = cv2.drawKeypoints(img, orb_kp, None)

#image = cv2.hconcat([img, orb_out])
image = orb_out

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()
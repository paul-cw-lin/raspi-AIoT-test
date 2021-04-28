import cv2
import argparse
'''
ap = argparse.ArgumentParser()
ap.add_argument('-i1', '--image1', required = True, help='first image')
ap.add_argument('-i2', '--image2', required=True, help='second image')
args = vars(ap.parse_args())

img1 = cv2.imread(args['image1.jpg'])
img2 = cv2.imread(args['image2.jpg'])
'''
img1 = cv2.imread('img26sr.jpg')
img2 = cv2.imread('img26d.jpg')

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x:x.distance)
img3 = cv2.drawMatches(img1, kp1,
                       img2, kp2,
                       matches[:20],
                       outImg=None,
                       flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

width, height, channel = img3.shape
ratio = float(width)/float(height)
img3 = cv2.resize(img3, (800, int(800*ratio)))

cv2.imshow('image', img3)
cv2.waitKey()
cv2.destroyAllWindows()
import cv2

o = cv2.imread('img7.jpg')
img = o

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 50, 150)
edged = cv2.dilate(edged, None, iterations=1)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

cnt = cv2.approxPolyDP(cnt, 30, True)

hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)

print('凸點數量 : {}' .format(len(hull)))
print('凹點數量 : {}' .format(len(defects)))

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    
    cv2.line(img, start, end, (0,255,0), 2)
    cv2.circle(img, far, 5, (0,0,255), -1)
    
cv2.imshow('frame', img)
cv2.waitKey()
cv2.destroyAllWindows()
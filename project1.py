import cv2 as cv
import numpy as np

image = cv.imread('input.jpg')
cv.imshow('Image', image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
cv.imshow('threshold', thresh)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
epsilon = 0.01*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)

cv.drawContours(image, [approx], -1, (0,255,0), 3)
cv.imshow('Find corner', image)

cv.imwrite('corner.jpg',image)

pts1 = np.float32([[367,236],[643,430],[59,592],[343,825]])
pts2 = np.float32([[0,0],[400,0],[0,600],[400,600]])

M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(image, M, (400, 600))

cv.imshow('Align crop',dst)

cv.waitKey(0)
cv.destroyAllWindows()
import numpy as np 
import cv2


img1=cv2.imread('first.png',cv2.IMREAD_COLOR)
img2=cv2.imread('second.png',cv2.IMREAD_COLOR)


add=img1 + img2
add2=cv2.add(img1,img2)
add3=cv2.addWeighted(img1,0.6,img2,0.4,0)
cv2.imshow('weighted', add3)
#cv2.imshow('addition',add2)
#cv2.imshow('addition', add)

#thresholding



cv2.waitKey(0)
cv2.destroyAllWindows()
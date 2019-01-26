import numpy as np
import cv2

img=cv2.imread('images.jpeg',cv2.IMREAD_COLOR)

#px=img[55,55]
#print px

#print(img.shape)
#print(img.size)
#print(img.dtype)

img[100:155,100:200]=[255,255,255]
region=img[1:50,1:50]
img[51:100,51:100]=region


cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

img=cv2.imread('/home/sid/Desktop/Axisstmt1.jpg', cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(img,(5,5),0)
th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

#cv2.Smooth(img,img2,smooth_type=CV.GAUSSIAN , param1=3 , param2=0 , param3=0 , param4=0)

import matplotlib.pyplot as plt

edges=cv2.Canny(th,10,50)

im2, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

#cv2.drawContours(img, contours, -1, (0,255,0), 3)



#cv2.imwrite('table_det.jpeg',edges)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

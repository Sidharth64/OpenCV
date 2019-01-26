import numpy as np 
import cv2

img=cv2.imread('image.jpeg',cv2.IMREAD_COLOR)

#cv2.line(img,(0,0),(65,64),(255,0,255),5)
#cv2.rectangle(img,(9,9),(90,89),(255,255,0),6)

#points=np.array([[23,21],[34,13],[58,78],[80,14]],np.int32)
#cv2.polylines(img,[points],True,(255,212,34),8)

font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Hello World!!',(20,1000),font,6,(0,0,253),14)

cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
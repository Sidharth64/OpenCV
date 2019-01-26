import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('image.jpeg',cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.show()
plt.xticks([0])
plt.yticks([])
cv2.imshow('output_img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('gray_img.jpeg',img)
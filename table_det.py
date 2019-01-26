import numpy as np 
import cv2
import tesserocr
from PIL import Image 
import math
from PIL import Image
import pytesseract
import os

img=cv2.imread('/home/sid/Desktop/table_det/Axisstmt1.jpg')
#cv2.imshow('color',img)
resized_image = cv2.resize(img, (800, 900))  
gray=cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY); 
#cv2.imshow('gray',gray)

th = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

#text = pytesseract.image_to_string(th)
#os.remove('try_image.jpg')
#print(text)

#cv2.imshow('thresh',th)
 
#th.shape[1]
scale=15 #parameter selection

horizontalsize=th.shape[1]/scale
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))

horizontal=cv2.erode(th, horizontalStructure) 
horizontal=cv2.dilate( horizontal, horizontalStructure)


verticalsize =th.shape[0]/scale #vertical.rows / scale;

verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize))

    
vertical=cv2.erode(th, verticalStructure)
vertical=cv2.dilate( vertical, verticalStructure)

#cv2.imshow("vertical", vertical)

#cv2.imshow("horizontal", horizontal)

mask = horizontal + vertical
#cv2.imshow("mask", mask)

joints=cv2.bitwise_and(horizontal, vertical)
#cv2.imshow("joints", joints)

img_1 , contours, hierarchy=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #, (0, 0));


#vector<vector<Point> > contours_poly( contours.size() )
#vector<Rect> boundRect( contours.size() )
#vector<Mat> rois;
contours_poly=[None for i in range(len(contours))]
boundRect=[None for i in range(len(contours))]
rois=[]
i=0
#print len(contours)

#cv2.drawContours(joints, contours, -1, (0,255,0), 3)
#cv2.imshow("title", joints)

while i<len(contours) :

    #find the area of each contour

    area = cv2.contourArea(contours[i])
  # filter individual lines of blobs that might exist and they do not represent a table
    
    
    if (area < 5) :
    	i=i+1 
        continue
    
    contours_poly[i]=cv2.approxPolyDP(contours[i], 3, True )
    
    x,y,w,h = cv2.boundingRect(contours_poly[i] )

    print x,y,w,h

  
    roi = joints[y:y+h,x:x+w]

    #joints_contours=cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    _,cnts,_ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ## filter by area
    
    s2 = 5.0
    xcnts = []
    for cnt in cnts:
        h=h+1
        if cv2.contourArea(cnt) <s2:
            xcnts.append(cnt)


    #print("Dots number: {}".format(len(xcnts)))

    if (len(xcnts)<=4):
        i=i+1
        continue

    rois.append(resized_image[y:y+h,x:x+w])

    cv2.rectangle( resized_image, (x,y), (x+w,y+h) , (0, 255, 0), 1, 8, 0 )
    i=i+1



j=0

print len(rois)


while j < len(rois):

    #cv2.imshow("roi", rois[j])
    gray=cv2.cvtColor(rois[j], cv2.COLOR_BGR2GRAY)
    th=cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

    #scale=15 #parameter selection

    horizontalsize=th.shape[1]/scale
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))

    horizontal=cv2.erode(th, horizontalStructure) 
    horizontal=cv2.dilate( horizontal, horizontalStructure)
    _, contours,_=cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #, (0, 0));
    
   
    lines = cv2.HoughLines(horizontal,1,np.pi/180,200)
    
    print len(lines)

    coord=[]
    
    for i in range(len(lines)):
        
        #print lines[i]
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        y0 = b * rho
        pt1 =  int(y0 + 1000*(a) )
        coord.append(pt1)

    j=j+1


coord=sorted(coord) 


#apply ocr on rois[j] using coord
rs_image = cv2.resize(rois[0][coord[3]:coord[4]+1,:], (1050,53))
cv2.imshow("not blur",rs_image)
#cv2.imwrite('trge.jpg',rs_image)
#rs_image = cv2.GaussianBlur(rs_image,(3,3),0)  
#cv2.imshow("blur",rs_image)

gray=cv2.cvtColor(rs_image,cv2.COLOR_BGR2GRAY)
#th = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
kernel = np.ones((1, 1), np.uint8)
#cv2.imshow('b',gray)

imgg = cv2.dilate(gray, kernel, iterations=1)
imgg = cv2.erode(imgg, kernel, iterations=1)
imgg = cv2.adaptiveThreshold(~imgg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

vertical=cv2.erode(imgg, verticalStructure)
vertical=cv2.dilate( vertical, verticalStructure)
imgg=imgg-vertical

cv2.imshow('fi',imgg)
blur = cv2.GaussianBlur(imgg,(3,3),0)

#imgg = cv2.morphologyEx(imgg, cv2.MORPH_GRADIENT, kernel)

#imgg = cv2.adaptiveThreshold(~imgg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

#cv2.imshow("contours", resized_image )

#cv2.imshow('first',blur)
text = pytesseract.image_to_string(imgg)
print text

#image = Image.open('trge.jpg')
#print (tesserocr.image_to_text(image))

#os.remove('try_image.jpg')
#print(text)

cv2.waitKey(0)

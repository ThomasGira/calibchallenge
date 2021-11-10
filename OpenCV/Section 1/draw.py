import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype='uint8')

#cv.imshow('Blank',blank)

#1. Paint the image a collor
#blank[200:300,400:500] = 0,255,0

#2. Draw Rectangle
#cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,255,0),thickness=-1)

#3. Draw Circle
#cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40,(0,0,255),thickness=3)

#4. Draw Line
# cv.line(blank,(blank.shape[1]//2,blank.shape[0]//2),(blank.shape[1]//4,blank.shape[0]//4),(255,0,255),thickness = 5)

#5. Write Text
cv.putText(blank,'ASS',(225,225),cv.FONT_HERSHEY_TRIPLEX,1.0,(14,69,8),thickness=2)
cv.imshow('Rect',blank)
cv.waitKey(0)
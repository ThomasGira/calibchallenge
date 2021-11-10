import cv2 as cv

img = cv.imread('Photos/cat.jpg')
cv.imshow('Cat',img)

#Grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)


# Blur
blur = cv.GaussianBlur(img, (11,11), cv.BORDER_DEFAULT)
cv.imshow('Blur',blur)

# Edge Cascade
canny = cv.Canny(img,125,175)
cv.imshow('Canny',canny)

# Dilating
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated',dilated)

# Eroding
eroded = cv.erode(dilated,(7,7), iterations=3)
cv.imshow('Eroded',eroded)

# Resize
resized = cv.resize(img,(500,500),interpolation = cv.INTER_CUBIC)
cv.imshow('Resized',resized) 

# Crop
cropped = img[50:200,200:300]
cv.imshow('Cropped',cropped)

cv.waitKey(0)
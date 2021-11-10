import cv2 as cv

img = cv.imread("Photos\cat.jpg")

cv.imshow('Cat', img)
cv.watikey(0)

#Reading Videos
capture = cv.VideoCapture('Videos\dog.mp4')
#capture = cv.VideoCapture(0) #Webcam

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
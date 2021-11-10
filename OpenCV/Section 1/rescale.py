import cv2 as cv

def rescaleFrame(frame,scale = 0.75):
    #Images, Videos and Live Videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame,dimensions, interpolation=cv.INTER_AREA)
    
def changeRes(width,height):
    #Live Video only
    capture.set(3,width)
    capture.set(4,height)

#capture = cv.VideoCapture('Videos\dog.mp4')
capture = cv.VideoCapture(0) #Webcam
t = 1
while True:
    isTrue, frame = capture.read()
    canny = cv.Canny(frame,125,175)
    cv.imshow('Canny',canny)
    cv.imshow('Video', frame)
    t += 10
    t = t%250
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
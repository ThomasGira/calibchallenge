import cv2
import matplotlib.pyplot as plt
import numpy as np

def findDirectionOfTravel(video):

    # Capture object
    cap = cv2.VideoCapture(video)

    if(not cap.isOpened()):
        print("Error opening video")
    
    #getting first frame 
    ret_last, frame_last = cap.read()
    frame_last = cv2.cvtColor(frame_last, cv2.COLOR_BGR2GRAY)
   
    h, w = np.shape(frame_last)

    # a mask to block most other cars and the hood of the car
    mask = 255*np.ones([h,w], dtype = "uint8")
    cv2.rectangle(mask, (2*w//5,2*h//5),(3*w//5,h//2),0,-1)
    cv2.rectangle(mask, (0,6*h//8), (w,h), 0, -1)
    cv2.imshow("",mask)

    while(cap.isOpened()):
        ret_curr, frame_curr = cap.read()

        if ret_curr:
            frame_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
           
           # I thought detecting edges first would help but it didn't
            # frame_last = cv2.GaussianBlur(frame_last,(3,3), sigmaX=0, sigmaY=0)
            # frame_curr = cv2.GaussianBlur(frame_curr,(3,3), sigmaX=0, sigmaY=0)
            
            # frame_last = cv2.Canny(image = frame_last, threshold1 = 100,
            #                                            threshold2 = 200)
            # frame_curr = cv2.Canny(image = frame_curr, threshold1 = 100,
                                                    #    threshold2 = 200)
            # 
# Full disclosure these were found by guess and check
            maxFeats = 500
            # maxFeats = 20
            scale = 2.1
            nlevels = 8
            edgeThresh = 1 
            firstLevel = 0
            WTA_K = 3
            scoreType = 2
            patchsize = 5
            fastThresh = 1
            orb = cv2.ORB_create(maxFeats,scale,nlevels,edgeThresh,firstLevel,
                    WTA_K, scoreType, patchsize,fastThresh)

            kps_last, des_last = orb.detectAndCompute(frame_last, mask)
            kps_curr, des_curr = orb.detectAndCompute(frame_curr, mask)
            
            # draw current and last keypoints
            annotatedFrame = cv2.drawKeypoints(frame_curr, kps_curr, np.array([]), (0,255,0),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            annotatedFrame = cv2.drawKeypoints(annotatedFrame, kps_last, np.array([]), (0,0,255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
            # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2)
            matcher = cv2.DescriptorMatcher_create(4)
            matches = matcher.match(des_last, des_curr, None)

            # # Sort based on the distance member of the items in matches
            # matches = sorted(matches,key = lambda x : x.distance)
            # matches = matches[1:100]
           
            # I expect key points to be close togeter as in the pixel coordinates
            # are close together because we want to find the same keypoints
            # accreoss two consecutive frames
            # Now, instead of sorting by hamming distance,
            # I'm going to try sorting by actual 2-norm pixel length
            # as in I want keypoints at pixels that are near by
            
            # ? am I sorting these correctly?
            L2Key = lambda x : abs(np.linalg.norm(np.asarray(kps_curr[x.trainIdx].pt) -
                                            np.asarray(kps_last[x.queryIdx].pt)))

            matches = sorted(matches, key = L2Key)
            matches = matches[1:50]
           
#             for i in range (9):
#                 print(matches[i].distance)
#                 print(matches[i].trainIdx)
#                 print(matches[i].queryIdx)
#                 print(matches[i].imgIdx)
#                 print()
            # these are the same memory address !!!! 
            # starwarsFrame = frame_curr
            # This is correct !!!
            starwarsFrame = np.array(frame_curr, copy = True)
            print(id(starwarsFrame))
            print(id(frame_curr))

            #draw lines for keypoint matches
            # print(type(kps_curr))
            # pt1 = np.zeros((1,2))
            # pt2 = np.zeros((1,2))
            for m in matches:
                #draw from (current to last) train to query (towards center)
                # pt1[0,:] = kps_curr[m.trainIdx].pt
                # print(type(kps_curr[m.trainIdx].pt[0]))
                # print(int(kps_curr[m.trainIdx].pt[0]))
                # print(pt1)
                # pt2[0,:] = kps_last[m.queryIdx].pt
                # starwarsFrame = cv2.line(starwarsFrame, pt1, pt2, (255,0,0))
                
                goodLine, slope, b, p1, p2 = lineFinder(kps_last[m.queryIdx].pt,
                                              kps_curr[m.trainIdx].pt,
                                              w,h)

                if goodLine:
                    starwarsFrame = cv2.line(starwarsFrame,p1,p2,(255,0,255))

#                 slope =  (int(kps_curr[m.trainIdx].pt[1]),int(kps_curr[m.trainIdx].pt[0])),
#                         (int(kps_last[m.queryIdx].pt[1]),int(kps_last[m.queryIdx].pt[0]))


                starwarsFrame = cv2.line(starwarsFrame,
                        (int(kps_curr[m.trainIdx].pt[0]),int(kps_curr[m.trainIdx].pt[1])),
                        (int(kps_last[m.queryIdx].pt[0]),int(kps_last[m.queryIdx].pt[1])),
                        (0,0,255))


            matchedFrame = np.array(cv2.drawMatches(frame_last, kps_last, frame_curr, 
                            kps_curr, matches, None), copy = True)

            cv2.imshow("starwars", starwarsFrame)

            cv2.imshow("matched", matchedFrame)

            cv2.imshow('Keypoints',annotatedFrame)

            cv2.waitKey(0)

            ret_last = ret_curr
            frame_last = np.array(frame_curr, copy = True)
    
            # Press q to quit the player
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        #### End of While

    cap.release()
    cv2.destroyAllWindows()


def lineFinder(queryPt, trainPt,w,h):
    # Line should go from train to query (current to last)
    x_2 = queryPt[0]
    y_2 = queryPt[1]

    x_1 = trainPt[0]
    y_1 = trainPt[1]
    if  (x_2 - x_1) == 0:
        return False, -1, -1, -1, -1

    # slope
    m = (y_2 - y_1)/(x_2 - x_1)
    
    # y intercept
    b = y_2 - m*x_2

    # two points outside of window
    p1 = (0,int(b))
    p2 = (w, int(m*w + b))
    goodLine = True

    return goodLine, m, b, p1, p2

def kpTrack():
    pass
    
    # maybe use cv2.drawMatches 

#playVid('calib_challenge/labeled/0.hevc')


findDirectionOfTravel('labeled/0.hevc')

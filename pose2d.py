import cv2 
import numpy as np


POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
nPoints = 15


def cut_frame(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    newimg = np.array(img)
    newimg = newimg[y-10:y+h+10, x-10:x+w+10]
    return newimg


def get_pose(frame, net):
    '''
        foots[] = 0 khong co chan
              = 1 chan phai co len
              = 2 chan trai co len
    '''

    foots = [0,0]
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    inHeight = 386
    inWidth = int(((aspect_ratio*inHeight)*8)//8)

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    points = []
    frameCopy = np.zeros(frame.shape)
    threshold = 0.2

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
        if prob > threshold :
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

    # # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            cv2.line(frameCopy, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frameCopy, points[partA], 1, (0, 0, 255), thickness=1, lineType=cv2.FILLED)

            if partA == 9 and partB == 10 :
                foots[0] = 1
            elif partA == 12 and partB == 13 :
                foots[1] = 1

    frameCopy = cv2.convertScaleAbs(frameCopy)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
    img = cut_frame(frameCopy)
    img = cv2.resize(img,(250, 700))
    img = np.array(img,np.float32).flatten()

    return img, foots
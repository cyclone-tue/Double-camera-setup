import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import aruco
import time

cap = cv2.VideoCapture(0)
cameraMatrix = np.array([
    [6.3392788734453904e+02, 0., 3.0808675953434488e+02],
    [0.,6.3040138666052906e+02, 2.0155498451453073e+02],
    [0., 0., 1. ]
    ])
distCoeffs = np.array([ 1.7802315026160684e-01, -2.2294989847251276e+00,
       -1.5298749543854606e-02, -3.3860921093235033e-04,
       1.3272266506874113e+01])

def threshold(image,values):
    low_h, low_s, low_v, high_h, high_s, high_v = values
    imconv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if (high_h < low_h):
        output1 = cv2.inRange(imconv, (low_h,low_s,low_v),(179, high_s, high_v))
        output2 = cv2.inRange(imconv, (0,low_s,low_v),(high_h, high_s, high_v))
        output = output1+output2        
    else:
        output=cv2.inRange(imconv, (low_h,low_s,low_v),(high_h, high_s, high_v))
    return output

def weirdFilter(image):
    kernel = np.ones((10,10),np.float32)
    return cv2.filter2D(image,-1,kernel)

while True:
    frame = cv2.imread('WIN_20190211_16_22_23_Pro.jpg',1)

    imred =     threshold(frame,[160, 32, 160, 10,  150, 255]) #red filter
    imgreen =   threshold(frame,[ 70, 32, 160, 90, 150, 255]) #green filter
    imblue =    threshold(frame,[110, 32, 160, 140, 150, 255]) #blue filter

    A1 = cv2.bitwise_and(weirdFilter(imred),weirdFilter(imgreen))
    A2 = cv2.bitwise_and(weirdFilter(imred),weirdFilter(imblue))
    A3 = cv2.bitwise_and(weirdFilter(imgreen),weirdFilter(imblue))
    
    B = cv2.bitwise_and(A1,imblue)
    G = cv2.bitwise_and(A2,imgreen)
    R = cv2.bitwise_and(A3,imred)

    RGB = cv2.bitwise_or(R,G)
    RGB = cv2.bitwise_or(RGB,B)

    combined = np.stack([imblue,imgreen,imred],axis=2)


    contours, hierarchy = cv2.findContours(RGB,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    CoordList=np.argwhere(RGB==255)
    points=np.array([[i,j] for [j,i] in CoordList])
    print(CoordList)
    print(points)


    if (len(points) > 20):
        outline = cv2.convexHull(points)
        approx=cv2.approxPolyDP(outline,10,True)

        #if (np.shape(approx)[0] == 4):
         #   rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(approx, 0.7, cameraMatrix, distCoeffs)

        cv2.drawContours(frame, [approx], 0, (255,0,0), 3)

    cv2.imshow("Frame+ellipse", frame)
    cv2.imshow("overlap",RGB)
    cv2.imshow("filtered",weirdFilter(combined))
    cv2.imshow("rgb",combined)   
    
 
    key = cv2.waitKey(1)
    if key == 27:
        break



cap.release()
cv2.destroyAllWindows()



import numpy as np
import cv2

cap = cv2.VideoCapture(0)

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
    _,frame = cap.read()

    imred =     threshold(frame,[160,32,160,10,150,255]) #red filter
    imgreen =   threshold(frame,[70,32,160,100,150,255]) #green filter
    imblue =    threshold(frame,[110,32,160,140,150,255]) #blue filter

    A1 = cv2.bitwise_and(weirdFilter(imred),weirdFilter(imgreen))
    A2 = cv2.bitwise_and(weirdFilter(imred),weirdFilter(imblue))
    A3 = cv2.bitwise_and(weirdFilter(imgreen),weirdFilter(imblue))
    B = cv2.bitwise_and(A1,imblue)
    G = cv2.bitwise_and(A2,imgreen)
    R = cv2.bitwise_and(A3,imred)

    RGB = cv2.bitwise_or(R,G)
    RGB = cv2.bitwise_or(RGB,B)

    combined = np.stack([imblue,imgreen,imred],axis=2)
    
    _, contours, hierarchy = cv2.findContours(RGB,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if (len(contours) > 20):
        points = np.vstack(contours)
        fit=cv2.fitEllipse(points)
        ellipse=cv2.ellipse(frame,fit,(255,0,0),5)

    cv2.imshow("rgb",combined)
    cv2.imshow("filtered",weirdFilter(combined))
    cv2.imshow("overlap",RGB)
    cv2.imshow("Frame+ellipse", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()


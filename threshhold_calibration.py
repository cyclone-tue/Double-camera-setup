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

def nothing(x):
    pass

cv2.namedWindow("balkjes")
cv2.createTrackbar("low_h","balkjes",0,179,nothing)
cv2.createTrackbar("low_s","balkjes",0,255,nothing)
cv2.createTrackbar("low_v","balkjes",0,255,nothing)
cv2.createTrackbar("high_h","balkjes",0,179,nothing)
cv2.createTrackbar("high_s","balkjes",0,255,nothing)
cv2.createTrackbar("high_v","balkjes",0,255,nothing)

while True:
    _,frame = cap.read()

    low_h = cv2.getTrackbarPos('low_h','balkjes')
    low_s = cv2.getTrackbarPos('low_s','balkjes')
    low_v = cv2.getTrackbarPos('low_v','balkjes')
    high_h = cv2.getTrackbarPos('high_h','balkjes')
    high_s = cv2.getTrackbarPos('high_s','balkjes')
    high_v = cv2.getTrackbarPos('high_v','balkjes')
    
    imblue = threshold(frame,[low_h,low_s,low_v,high_h, high_s, high_v]) #blue filter

    cv2.imshow("blue", imblue)
    cv2.imshow("Frame+ellipse", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()


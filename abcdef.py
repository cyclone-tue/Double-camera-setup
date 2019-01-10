import numpy as np
import cv2
import argparse

cap = cv2.VideoCapture(0)

def threshold(input,values):
    low_h, low_s, low_v, high_h, high_s, high_v = values
    inputconv=cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    output=cv2.inRange(inputconv, (low_h,low_s,low_v),(high_h, high_s, high_v))
    return output
    

while True:
    _, frame = cap.read()

    im = threshold(frame,[150,100,200,255,150,255]) #red filter
    im2, contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours)>=5):
        print(np.vstack(contours))
        fit=cv2.fitEllipse(np.vstack(contours))
        ellipse=cv2.ellipse(frame,fit,(255,0,0),5)

    cv2.imshow("treshold", im)
    cv2.imshow("Frame+ellpise", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()


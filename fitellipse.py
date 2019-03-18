import numpy as np
import cv2
from numpy import linalg as LA

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

def overlap(imred,imgreen,imblue):
    A1 = cv2.bitwise_and(weirdFilter(imred), weirdFilter(imgreen))
    A2 = cv2.bitwise_and(weirdFilter(imred), weirdFilter(imblue))
    A3 = cv2.bitwise_and(weirdFilter(imgreen), weirdFilter(imblue))
    B = cv2.bitwise_and(A1, imblue)
    G = cv2.bitwise_and(A2, imgreen)
    R = cv2.bitwise_and(A3, imred)

    RGB = cv2.bitwise_or(R, G)
    RGB = cv2.bitwise_or(RGB, B)
    return RGB

while True:
    _,frame = cap.read()

    imred =     threshold(frame,[160,32,160,10,150,255]) #red filter
    imgreen =   threshold(frame,[70,32,160,100,150,255]) #green filter
    imblue =    threshold(frame,[110,32,160,140,150,255]) #blue filter

    RGB = overlap(imred,imgreen,imblue)

    combined = np.stack([imblue,imgreen,imred],axis=2)
    
    contours, hierarchy = cv2.findContours(RGB,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if (len(contours) > 20):
        points = np.vstack(contours)
        fit=cv2.fitEllipse(points)
        ellipse=cv2.ellipse(frame,fit,(255,0,0),5)
        (xc, yc), (a,b), theta = fit        # Aanpassen indien nodig

        A = a**2*np.sin(theta)**2+b**2*np.cos(theta)**2
        B = 2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
        C = a**2*np.cos(theta)**2+b**2*np.sin(theta)**2
        D = -2*A*xc-2*B*yc
        E = -B*xc-2*C*yc
        F = A*xc**2+B*xc*yc+C*yc**2-a**2*b**2

        # print([A,B,C,D,E,F])

        Q=np.array([[A,B,D],[B,C,E],[D,E,F]])
        w,v = LA.eig(Q)

        print(w)
        print(v)



    cv2.imshow("rgb",combined)
    cv2.imshow("filtered",weirdFilter(combined))
    cv2.imshow("overlap",RGB)
    cv2.imshow("Frame+ellipse", frame)




    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()


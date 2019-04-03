import numpy as np
import cv2
from cv2 import aruco
from numpy import linalg as LA
import itertools

dictionary = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
cap = cv2.VideoCapture(0)


# Tunable
alpha = 0
S1 = -1
S2 = -1
S3 = -1

f=6.3392788734453904e+02
r=0.375

markerLength = 0.141 #in meters

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

def getEllipseParams(fit):
    (xc, yc), (a, b), theta = fit

    a=a/2
    b=b/2
    theta = theta*np.pi/180

    A = a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2
    B = (b ** 2 - a ** 2) * np.sin(theta) * np.cos(theta)
    C = a ** 2 * np.cos(theta) ** 2 + b ** 2 * np.sin(theta) ** 2
    D = -A * xc - B * yc / 2
    E = -B * xc / 2 - C * yc
    F = A * xc ** 2 + B * xc * yc + C * yc ** 2 - a ** 2 * b ** 2

    return A,B,C,D,E,F

def nothing(x):
    pass


cv2.namedWindow("balkjes")
cv2.createTrackbar("alpha","balkjes",0,7,nothing)
cv2.createTrackbar("xc","balkjes",0,640,nothing)
cv2.createTrackbar("yc","balkjes",0,480,nothing)
cv2.createTrackbar("a","balkjes",1,480,nothing)
cv2.createTrackbar("b","balkjes",1,480,nothing)
cv2.createTrackbar("theta","balkjes",0,360,nothing)

cv2.namedWindow("SJES")
cv2.createTrackbar("S1","SJES",0,2,nothing)
cv2.createTrackbar("S2","SJES",0,2,nothing)
cv2.createTrackbar("S3","SJES",0,2,nothing)

while True:
    alpha = cv2.getTrackbarPos("alpha","balkjes")
    xc = cv2.getTrackbarPos("xc","balkjes")
    yc = cv2.getTrackbarPos("yc","balkjes")
    a = cv2.getTrackbarPos("a","balkjes")
    b = cv2.getTrackbarPos("b","balkjes")
    theta = cv2.getTrackbarPos("theta","balkjes")
    S1 = -1+cv2.getTrackbarPos("S1","SJES")
    S2 = -1+cv2.getTrackbarPos("S2","SJES")
    S3 = -1+cv2.getTrackbarPos("S3","SJES")
    
    frame = cv2.imread("WIN_20190318_17_37_44_Pro.jpg")

    imred =     threshold(frame,[160,32,160,10,150,255]) #red filter
    imgreen =   threshold(frame,[70,32,160,100,150,255]) #green filter
    imblue =    threshold(frame,[110,32,160,140,150,255]) #blue filter

    RGB = overlap(imred,imgreen,imblue)

    combined = np.stack([imblue,imgreen,imred],axis=2)
    
    contours, hierarchy = cv2.findContours(RGB,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #random test
    corners, ids, _ = aruco.detectMarkers(frame,dictionary)
    aruco.drawDetectedMarkers(frame,corners,ids)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)
        for (rvec, tvec) in zip(rvecs,tvecs):
            aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
            
    if (len(contours) > 20):
        points = np.vstack(contours)
        #set custom ellipse
        fit= cv2.fitEllipse(points)  #(xc, yc), (a, b), theta

        #draw ellipse
        ellipse=cv2.ellipse(frame,fit,(255,0,0),5)

        A, B, C, D, E, F = getEllipseParams(fit)
        Qe = np.array([[A,B,-D/f],[B,C,-E/f],[-D/f,-E/f,F/f**2]])
        w,V = LA.eig(Qe)

        #making sure the eigenvalue condition is satified
        for p in itertools.permutations([0,1,2]):
            pw = l1,l2,l3 = [w[i] for i in p]  #permutation applied to w
            if (l1*l2 > 0 and l1*l3<0 and abs(l1)>=abs(l2)):
                w=pw
                V=np.transpose(np.array([V[:,i] for i in p])) #permutation applied to V
                break

        l1,l2,l3= w

        g = np.sqrt((l2-l3)/(l1-l3))
        h = np.sqrt((l1-l2)/(l1-l3))

        Rc = V.dot(np.array([[g*np.cos(alpha), S1*g*np.sin(alpha), S2*h],
                           [np.sin(alpha), -S1*np.cos(alpha),0],
                           [S1*S2*h*np.cos(alpha), S2*h*np.sin(alpha),-S1*g]]))

        z0=S3*l2*r/np.sqrt(-l1*l3)
        Cvector = z0*V.dot(np.array([[S2*l3/l2*h], [0], [-S1*l1/l2*g]]))
        
        Nvector=V*np.array([S2*h, 0, -S1*g])

        rvec,_ = cv2.Rodrigues(Rc)
        tvec = np.array([[0.],[0.],[3.]])
        
        cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
        #cv2.circle(frame, (int(xc),int(yc)), 10, (255,0,0))

    #cv2.imshow("rgb",combined)
    #cv2.imshow("filtered",weirdFilter(combined))
    cv2.imshow("overlap",RGB)
    cv2.imshow("Frame+ellipse", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()


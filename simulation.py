import numpy as np
import cv2
from cv2 import aruco
from numpy import linalg as LA
import itertools

cameraMatrix = np.array([
    [6.e+02, 0., 768.],
    [0.,6.e+02, 432.],
    [0., 0., 1. ]
])

#np.array([
#    [6.3392788734453904e+02, 0., 3.0808675953434488e+02],
#    [0.,6.3040138666052906e+02, 2.0155498451453073e+02],
#    [0., 0., 1. ]
#])

distCoeffs = np.array([0.,0.,0.,0.,0.])

#np.array([ 1.7802315026160684e-01, -2.2294989847251276e+00,
#       -1.5298749543854606e-02, -3.3860921093235033e-04,
#       1.3272266506874113e+01])

rvec = np.array([[0.],[0.],[0.]])
tvec = np.array([[0.],[0.5],[5.]])

def create_grid(rows, cols, length):
    grid = np.zeros([rows,cols,3])
    for i in range(rows):
        for j in range(cols):
            grid[i][j]=[i*length-(rows-1)*length/2,0,j*length-(cols-1)*length/2]
    return grid

def create_hoop(r, px,py,pz):
    hoop = np.zeros([41,3])
    for i in range(41):
        hoop[i]=[px+r*np.cos(i*np.pi/20),py+r*np.sin(i*np.pi/20),pz]
    return hoop
    

def project_grid(grid, rvec, tvec, cameraMatrix, distCoeffs):
    return np.array([
        cv2.projectPoints(row, rvec, tvec, cameraMatrix, distCoeffs)[0]
        for row in grid])

def draw_grid(grid, img, rvec, tvec, cameraMatrix, distCoeffs):
    pgrid = project_grid(grid, rvec, tvec, cameraMatrix, distCoeffs)
    pgrid = pgrid.astype(np.int64)
    for i in range(grid.shape[0]): #rows
        for j in range(grid.shape[1]): #cols
            if i!=0:
                pt1= tuple(pgrid[i][j][0])
                pt2= tuple(pgrid[i-1][j][0])
                cv2.line(img, pt1, pt2, (0,255,0),1)
            if j!=0:
                pt1= tuple(pgrid[i][j][0])
                pt2= tuple(pgrid[i][j-1][0])
                cv2.line(img, pt1, pt2, (0,255,0),1)

def draw_hoop(hoop, img, rvec, tvec, cameraMatrix, distCoeffs):
    phoop = cv2.projectPoints(hoop, rvec, tvec, cameraMatrix, distCoeffs)[0]
    phoop = phoop.astype(np.int64)
    for i in range(hoop.shape[0]): #number of pionts
        if i!=0:
            pt1 = tuple(phoop[i][0])
            pt2 = tuple(phoop[i-1][0])
            cv2.line(img, pt1, pt2, (0,255,0),1)

grid = create_grid(10,10,0.3)
hoop = create_hoop(1,0,-2,0)

while True:
    #using screen resolution of 1536x864
    frame1 = np.zeros((864,1536,3),dtype=np.uint8) #shape = (480, 640, 3)
    frame2 = np.zeros((864,1536,3),dtype=np.uint8) #shape = (480, 640, 3)    
    
    draw_grid(grid, frame1, rvec, tvec, cameraMatrix, distCoeffs)
    draw_hoop(hoop, frame1, rvec, tvec, cameraMatrix, distCoeffs)
    
    cv2.imshow("simulation", frame1)
    

    control = cv2.waitKeyEx()
    print(control)
    if control == 27:
        break
    if control == 119:              #w          FORWARD
        tvec[2] = tvec[2] - 0.1
    if control == 115:              #s          BACKWARD
        tvec[2] = tvec[2] + 0.1
    if control == 100:              #d          RIGHT
        tvec[0] = tvec[0] - 0.1
    if control == 97:               #a          LEFT
        tvec[0] = tvec[0] + 0.1
    if control == 2490368:          #uparrow    UP
        tvec[1] = tvec[1] + 0.1
    if control == 2621440:          #downarrow  DOWN
        tvec[1] = tvec[1] - 0.1
        

cv2.destroyAllWindows()

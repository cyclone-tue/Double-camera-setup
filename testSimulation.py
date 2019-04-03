import numpy as np
import cv2
from cv2 import aruco
from numpy import linalg as LA
import itertools

class Camera:
    def __init__(self, rMat, pos, cameraMatrix, distCoeffs):
        self.rMat = rMat    # wrt world frame
        self.pos = pos      # wrt world frame
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

    # projects 3d points from world frame to 2d camera image
    def project(self, points):        
        tvec = -np.dot(np.transpose(self.rMat),self.pos)
        rvec = cv2.Rodrigues(np.transpose(self.rMat))[0]
        return cv2.projectPoints(points, rvec, tvec, self.cameraMatrix, self.distCoeffs)[0].astype(np.int64)

# Calculates Rotation Matrix given euler angles.
def RotationMatrix(theta):
    R_x = np.array([[1,0,0],
                    [0,np.cos(theta[0]),-np.sin(theta[0])],
                    [0,np.sin(theta[0]), np.cos(theta[0])]
                    ])  
    R_y = np.array([[np.cos(theta[1]),0,np.sin(theta[1])],
                    [0,1,0],
                    [-np.sin(theta[1]),0,np.cos(theta[1])]
                    ])                 
    R_z = np.array([[np.cos(theta[2]),-np.sin(theta[2]),0],
                    [np.sin(theta[2]), np.cos(theta[2]),0],
                    [0,0,1]
                    ])                 
    R = np.dot(R_z, np.dot( R_y, R_x )) 
    return R

def create_grid(rows, cols, length):
    grid = np.zeros([rows,cols,3])
    for i in range(rows):
        for j in range(cols):
            grid[i][j]=[i*length-(rows-1)*length/2,0,j*length-(cols-1)*length/2]
    return grid

def create_hoop(r, px,py,pz):
    hoop = np.zeros([21,3])
    for i in range(21):
        hoop[i]=[px+r*np.cos(i*np.pi/10),py+r*np.sin(i*np.pi/10),pz]
    return hoop
    

def project_grid(grid, cam):
    return np.array([
        cam.project(row)
        for row in grid])

def draw_grid(grid, img, cam):
    pgrid = project_grid(grid, cam)
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

def draw_hoop(hoop, img, cam):
    phoop = cam.project(hoop)
    for i in range(hoop.shape[0]): #number of pionts
        if i!=0:
            pt1 = tuple(phoop[i][0])
            pt2 = tuple(phoop[i-1][0])
            cv2.line(img, pt1, pt2, (0,255,0),1)


def update_orientation(event, x, y, flags, params):
    global xi,yi, dragging, cam1
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        xi,yi = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            #print(x-xi,y-yi)
            yaw = np.pi*(x-xi)/1536
            pitch = np.pi*(y-yi)/(2*864)
            cam1.rMat = RotationMatrix([pitch,-yaw,0])
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False


rvec = np.array([[0.],[0.],[0.]])
tvec = np.array([[0.],[0.5],[5.]])

dragging = False
xi,yi=-1,-1

cam1 = Camera(
    rMat = RotationMatrix([0,0,0]),
    pos = np.array([[0.],[0.5],[5.]]),
    cameraMatrix = np.array([[6.e+02, 0., 768.], [0.,6.e+02, 432.], [0., 0., 1. ]]),
    distCoeffs = np.array([0.,0.,0.,0.,0.])
    )

grid = create_grid(10,10,0.3)
hoop = create_hoop(1,0,2,0)
cv2.namedWindow('simulation')
cv2.setMouseCallback('simulation', update_orientation)


while True:
    #using screen resolution of 1536x864
    frame1 = np.zeros((864,1536,3),dtype=np.uint8) #shape = (480, 640, 3)
    frame2 = np.zeros((864,1536,3),dtype=np.uint8) #shape = (480, 640, 3)    
    
    draw_grid(grid, frame1, cam1)
    draw_hoop(hoop, frame1, cam1)

    # fit = cv2.fitEllipse(cam1.project(hoop))
    # cv2.ellipse(frame1,fit,(255,0,0),5)

    cv2.imshow("simulation", frame1)
    
    control = cv2.waitKeyEx(1)
    if control == 27:
        break
    if control == 119:              #w          FORWARD
        cam1.pos[2] = cam1.pos[2] - 0.1
    if control == 115:              #s          BACKWARD
        cam1.pos[2] = cam1.pos[2] + 0.1
    if control == 100:              #d          RIGHT
        cam1.pos[0] = cam1.pos[0] - 0.1
    if control == 97:               #a          LEFT
        cam1.pos[0] = cam1.pos[0] + 0.1
    if control == 2490368:          #uparrow    UP
        cam1.pos[1] = cam1.pos[1] + 0.1
    if control == 2621440:          #downarrow  DOWN
        cam1.pos[1] = cam1.pos[1] - 0.1
        

cv2.destroyAllWindows()

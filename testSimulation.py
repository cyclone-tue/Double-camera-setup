import numpy as np
import cv2
#from cv2 import aruco
from numpy import linalg as LA
import itertools

# Calculates Rotation Matrix given euler angles.
def RotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

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


class Camera:
    def __init__(self, rMat, pos, cameraMatrix, distCoeffs):
        self.yaw = 0        # wrt world frame
        self.pitch = 0
        self.roll = 0
        self.rMat = rMat
        self.pos = pos      # wrt world frame
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

    def update(self):
        self.rMat = RotationMatrix([self.pitch, self.yaw, self.roll])

    # projects 3d points from world frame to 2d camera image
    def project(self, points):
        # points = np.array(list(filter(lambda x: x[2] > 0, points)))

        # y-axis is used as projection axis
        M = np.dot(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), self.rMat)

        tvec = -np.dot(np.transpose(M), self.pos)
        rvec = cv2.Rodrigues(np.transpose(M))[0]
        return cv2.projectPoints(points, rvec, tvec, self.cameraMatrix, self.distCoeffs)[0].astype(np.int64)


def create_grid(rows, cols, length):
    grid = np.zeros([rows,cols,3])
    for i in range(rows):
        for j in range(cols):
            grid[i][j]=[i*length-(rows-1)*length/2, j*length-(cols-1)*length/2,0]
    return grid

def create_hoop(r, px,py,pz):
    hoop = np.zeros([41,3])
    for i in range(41):
        hoop[i] = [px+r*np.cos(i*np.pi/20), py, pz+r*np.sin(i*np.pi/20)]
    return hoop
    

def project_grid(grid, cam):
    return np.array([
        cam.project(row)
        for row in grid])

def draw_grid(grid, img, cam):
    pgrid = project_grid(grid, cam)
    for i in range(pgrid.shape[0]):         # rows
        for j in range(pgrid.shape[1]):     # cols
            if i != 0:
                pt1 = tuple(pgrid[i][j][0])
                pt2 = tuple(pgrid[i-1][j][0])
                cv2.line(img, pt1, pt2, (0,255,0),1)
            if j != 0:
                pt1 = tuple(pgrid[i][j][0])
                pt2 = tuple(pgrid[i][j-1][0])
                cv2.line(img, pt1, pt2, (0,255,0),1)

def draw_hoop(hoop, img, cam):
    phoop = cam.project(hoop)
    for i in range(phoop.shape[0]):     # number of points
        if i != 0:
            pt1 = tuple(phoop[i][0])
            pt2 = tuple(phoop[i-1][0])
            cv2.line(img, pt1, pt2, (0, 255, 0), 1)


def update_orientation(event, x, y, flags, params):
    global xi, yi, dragging, cam1
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        xi, yi = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            cam1.yaw += -np.pi*(x-xi)/1536
            cam1.pitch += np.pi*(y-yi)/(2*864)
            xi, yi = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False


dragging = False
xi, yi = -1, -1

cam1 = Camera(
    rMat=np.identity(3),
    pos=np.array([[0.], [-5.], [2.]]),
    cameraMatrix=np.array([[6.e+02, 0., 768.], [0., 6.e+02, 432.], [0., 0., 1.]]),
    distCoeffs=np.array([0., 0., 0., 0., 0.])
    )

# Tunable
alpha = 0
S1 = -1
S2 = -1
S3 = -1

f=600
r=0.375

grid = create_grid(10, 10, 0.3)
hoop = create_hoop(1, px=0, py=0, pz=2)
#hoop2 = create_hoop(1, px=3, py=0, pz=2)

cv2.namedWindow('simulation')
cv2.setMouseCallback('simulation', update_orientation)

fit_ellipse = False

while True:
    cam1.update()
    # using screen resolution of 1536x864
    frame1 = np.zeros((864,1536,3), dtype = np.uint8)  # cv2.imread("images.jpg")
    frame2 = np.zeros((864, 1536, 3), dtype=np.uint8)  # shape = (480, 640, 3)

    draw_grid(grid, frame1, cam1)
    draw_hoop(hoop, frame1, cam1)
    #draw_hoop(hoop2, frame1, cam1)



    if fit_ellipse:
        fit = cv2.fitEllipse(cam1.project(hoop))
        cv2.ellipse(frame1, fit, (255, 0, 0), 5)

        A, B, C, D, E, F = getEllipseParams(fit)
        Qe = np.array([[A, B, -D / f], [B, C, -E / f], [-D / f, -E / f, F / f ** 2]])
        w, V = LA.eig(Qe)

        # making sure the eigenvalue condition is satified
        for p in itertools.permutations([0, 1, 2]):
            pw = l1, l2, l3 = [w[i] for i in p]  # permutation applied to w
            if (l1 * l2 > 0 and l1 * l3 < 0 and abs(l1) >= abs(l2)):
                w = pw
                V = np.transpose(np.array([V[:, i] for i in p]))  # permutation applied to V
                break

        l1, l2, l3 = w

        g = np.sqrt((l2 - l3) / (l1 - l3))
        h = np.sqrt((l1 - l2) / (l1 - l3))

        Rc = V.dot(np.array([[g * np.cos(alpha), S1 * g * np.sin(alpha), S2 * h],
                             [np.sin(alpha), -S1 * np.cos(alpha), 0],
                             [S1 * S2 * h * np.cos(alpha), S2 * h * np.sin(alpha), -S1 * g]]))

        z0 = S3 * l2 * r / np.sqrt(-l1 * l3)
        Cvector = z0 * V.dot(np.array([[S2 * l3 / l2 * h], [0], [-S1 * l1 / l2 * g]]))

        Nvector = V * np.array([S2 * h, 0, -S1 * g])

        rvec, _ = cv2.Rodrigues(Rc)
        tvec = np.array([[0.], [0.], [3.]])

        cv2.aruco.drawAxis(frame1, cam1.cameraMatrix, cam1.distCoeffs, rvec, tvec, 0.1)


    cv2.rectangle(frame1, (10, 10), (310, 320), (0, 0, 0), -1)
    cv2.rectangle(frame1, (10, 10), (310, 320), (0, 255, 0), 1)

    cv2.putText(frame1, "position:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    cv2.putText(frame1, "x={:.2f}".format(cam1.pos[0][0]), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    cv2.putText(frame1, "y={:.2f}".format(cam1.pos[1][0]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    cv2.putText(frame1, "z={:.2f}".format(cam1.pos[2][0]), (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    cv2.putText(frame1, "orientation:", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    cv2.putText(frame1, "yaw={:.2f}".format(cam1.yaw*180/np.pi), (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    cv2.putText(frame1, "pitch={:.2f}".format(cam1.pitch*180/np.pi), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    cv2.putText(frame1, "roll={:.2f}".format(cam1.roll*180/np.pi), (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    control = cv2.waitKeyEx(1)

    if control == 27:
        break
    if control == 102:
        fit_ellipse = not fit_ellipse
    if control == 119:              # w          FORWARD
        cam1.pos[1] = cam1.pos[1] + 0.1
    if control == 115:              # s          BACKWARD
        cam1.pos[1] = cam1.pos[1] - 0.1
    if control == 100:              # d          RIGHT
        cam1.pos[0] = cam1.pos[0] + 0.1
    if control == 97:               # a          LEFT
        cam1.pos[0] = cam1.pos[0] - 0.1
    if control == 2490368:          # up arrow   UP
        cam1.pos[2] = cam1.pos[2] + 0.1
    if control == 2621440:          # down arrow DOWN
        cam1.pos[2] = cam1.pos[2] - 0.1
    if control == 2555904:          # right arrow
        cam1.roll += 0.01
    if control == 2424832:          # left arrow
        cam1.roll -= 0.01
    print (control)

    cv2.imshow("simulation", frame1)

cv2.destroyAllWindows()

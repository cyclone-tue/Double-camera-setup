import numpy as np
import cv2
from cv2 import aruco
from numpy import linalg as LA
import itertools

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


class Camera:
    def __init__(self, rMat, pos, cameraMatrix, distCoeffs):
        self.rMat = rMat  # wrt world frame
        self.pos = pos  # wrt world frame
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

    # projects 3d points from world frame to 2d camera image
    def project(self, points):
        tvec = -np.dot(np.transpose(self.rMat), self.pos)
        rvec = cv2.Rodrigues(np.transpose(self.rMat))[0]
        return cv2.projectPoints(points, rvec, tvec, self.cameraMatrix, self.distCoeffs)[0].astype(np.int64)


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


def create_grid(rows, cols, length):
    grid = np.zeros([rows, cols, 3])
    for i in range(rows):
        for j in range(cols):
            grid[i][j] = [i * length - (rows - 1) * length / 2, 0, j * length - (cols - 1) * length / 2]
    return grid


def create_hoop(r, px, py, pz):
    hoop = np.zeros([21, 3])
    for i in range(21):
        hoop[i] = [px + r * np.cos(i * np.pi / 10), py + r * np.sin(i * np.pi / 10), pz]
    return hoop


def project_grid(grid, cam):
    return np.array([
        cam.project(row)
        for row in grid])


def draw_grid(grid, img, cam):
    pgrid = project_grid(grid, cam)
    for i in range(grid.shape[0]):  # rows
        for j in range(grid.shape[1]):  # cols
            if i != 0:
                pt1 = tuple(pgrid[i][j][0])
                pt2 = tuple(pgrid[i - 1][j][0])
                cv2.line(img, pt1, pt2, (0, 255, 0), 1)
            if j != 0:
                pt1 = tuple(pgrid[i][j][0])
                pt2 = tuple(pgrid[i][j - 1][0])
                cv2.line(img, pt1, pt2, (0, 255, 0), 1)


def draw_hoop(hoop, img, cam):
    phoop = cam.project(hoop)
    for i in range(hoop.shape[0]):  # number of pionts
        if i != 0:
            pt1 = tuple(phoop[i][0])
            pt2 = tuple(phoop[i - 1][0])
            cv2.line(img, pt1, pt2, (0, 255, 0), 1)


def update_orientation(event, x, y, flags, params):
    global xi, yi, dragging, cam1
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        xi, yi = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            # print(x-xi,y-yi)
            yaw = np.pi * (x - xi) / 1536
            pitch = np.pi * (y - yi) / (2 * 864)
            cam1.rMat = RotationMatrix([pitch, -yaw, 0])
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

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

cv2.namedWindow("SJES")
cv2.createTrackbar("S1","SJES",0,2,nothing)
cv2.createTrackbar("S2","SJES",0,2,nothing)
cv2.createTrackbar("S3","SJES",0,2,nothing)

rvec = np.array([[0.], [0.], [0.]])
tvec = np.array([[0.], [0.5], [5.]])

dragging = False
xi, yi = -1, -1

cam1 = Camera(
    rMat=RotationMatrix([0, 0, 0]),
    pos=np.array([[0.], [0.5], [5.]]),
    cameraMatrix=np.array([[6.e+02, 0., 768.], [0., 6.e+02, 432.], [0., 0., 1.]]),
    distCoeffs=np.array([0., 0., 0., 0., 0.])
)

grid = create_grid(10, 10, 0.3)
hoop = create_hoop(1, 0, 2, 0)
cv2.namedWindow('simulation')
cv2.setMouseCallback('simulation', update_orientation)

while True:
    # using screen resolution of 1536x864
    frame1 = np.zeros((864, 1536, 3), dtype=np.uint8)  # shape = (480, 640, 3)
    frame2 = np.zeros((864, 1536, 3), dtype=np.uint8)  # shape = (480, 640, 3)

    draw_grid(grid, frame1, cam1)
    draw_hoop(hoop, frame1, cam1)

    fit = cv2.fitEllipse(cam1.project(hoop))  # (xc, yc), (a, b), theta
    cv2.ellipse(frame1, fit, (255, 0, 0), 5)


    control = cv2.waitKeyEx(1)
    if control == 27:
        break
    if control == 119:  # w          FORWARD
        cam1.pos[2] = cam1.pos[2] - 0.1
    if control == 115:  # s          BACKWARD
        cam1.pos[2] = cam1.pos[2] + 0.1
    if control == 100:  # d          RIGHT
        cam1.pos[0] = cam1.pos[0] - 0.1
    if control == 97:  # a          LEFT
        cam1.pos[0] = cam1.pos[0] + 0.1
    if control == 2490368:  # uparrow    UP
        cam1.pos[1] = cam1.pos[1] + 0.1
    if control == 2621440:  # downarrow  DOWN
        cam1.pos[1] = cam1.pos[1] - 0.1

    ####################   FitEllipse   ###########################
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

    S1 = -1 + cv2.getTrackbarPos("S1", "SJES")
    S2 = -1 + cv2.getTrackbarPos("S2", "SJES")
    S3 = -1 + cv2.getTrackbarPos("S3", "SJES")


    Rc = V.dot(np.array([[g * np.cos(alpha), S1 * g * np.sin(alpha), S2 * h],
                         [np.sin(alpha), -S1 * np.cos(alpha), 0],
                         [S1 * S2 * h * np.cos(alpha), S2 * h * np.sin(alpha), -S1 * g]]))

    t = np.array([-S2*S3*r*np.cos(alpha)*np.sqrt((l1-l2)*(l2-l3)/(-l1)/l3),-S1*S2*S3*r*np.sin(alpha)*np.sqrt((l1-l2)*(l2-l3)/(-l1)/l3),S3*r*l2/np.sqrt(-l1*l3)])

    z0 = S3 * l2 * r / np.sqrt(-l1 * l3)
    Cvector = z0 * V.dot(np.array([[S2 * l3 / l2 * h], [0], [-S1 * l1 / l2 * g]]))

    Nvector = V * np.array([S2 * h, 0, -S1 * g])

    rvec, _ = cv2.Rodrigues(Rc)
    tvec = t

    print(t)

    cv2.aruco.drawAxis(frame1, cameraMatrix, np.array([0., 0., 0., 0., 0.]), rvec, tvec, 0.1)

    cv2.imshow("simulation", frame1)

cv2.destroyAllWindows()

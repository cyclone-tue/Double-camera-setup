import numpy as np
import cv2
#from cv2 import aruco
from numpy import linalg as LA
import itertools

import simulation as sim

def getEllipseParams(fit):
    (xc, yc), (a, b), theta = fit


    xc=xc-768
    yc=-yc+432
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


def update_orientation(event, x, y, flags, params):
    global xi, yi, dragging, cam1
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        xi, yi = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            dcam.rotate(-np.pi*(x-xi)/1536, np.pi*(y-yi)/(2*864), 0)
            xi, yi = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False


dragging = False
xi, yi = -1, -1

cam1 = sim.Camera(
    rMat=np.identity(3),
    pos=np.array([[-1], [-5.], [2.]]),
    cameraMatrix=np.array([[6.e+02, 0., 1.5*320.], [0., 6.e+02, 1.5*240.], [0., 0., 1.]]),
    distCoeffs=np.array([0., 0., 0., 0., 0.])
    )

cam2 = sim.Camera(
    rMat=np.identity(3),
    pos=np.array([[1], [-5.], [2.]]),
    cameraMatrix=np.array([[6.e+02, 0., 1.5*320.], [0., 6.e+02, 1.5*240.], [0., 0., 1.]]),
    distCoeffs=np.array([0., 0., 0., 0., 0.])
    )

dcam = sim.DoubleCamera(np.array([[0], [-5.], [2.]]), cam1, cam2)

# Tunable
alpha = 0
S1 = -1
S2 = 1
S3 = 1

f=600
r=0.375

grid = sim.create_grid(10, 10, 0.3)
hoop = sim.create_hoop(1, px=0, py=0, pz=0)
square = sim.create_square(1, px=0, py=0, pz=0)
#hoop2 = create_hoop(1, px=3, py=0, pz=2)


cv2.namedWindow('simulation')
cv2.setMouseCallback('simulation', update_orientation)

fit_ellipse = False

##### To find h ####
#basevec = np.zeros([1, 3])
#basevec[0][0] = 0.5*np.sqrt(2)
#basevec[0][1] = 0
#basevec[0][2] = 0.5*np.sqrt(2)
#h = cam1.project(basevec)
#h = h - [768, 432]
#print(basevec)
#print("projection is")
#print(h)


######## S jes instelbaar #######################
def nothing(x):
    pass


cv2.namedWindow("SJES")
cv2.createTrackbar("S1", "SJES", 0, 2, nothing)
cv2.createTrackbar("S2", "SJES", 0, 2, nothing)
cv2.createTrackbar("S3", "SJES", 0, 2, nothing)
##################################################


while True:
    dcam.update()
    # using screen resolution of 1536x864
    frame1 = np.zeros((720, 960, 3), dtype=np.uint8)  # cv2.imread("images.jpg")
    frame2 = np.zeros((720, 960, 3), dtype=np.uint8)  # shape = (480, 640, 3)

    sim.draw_grid(grid, frame1, cam1)
    sim.draw_hoop(hoop, frame1, cam1)
    sim.draw_square(square, frame1, cam1)

    sim.draw_grid(grid, frame2, cam2)
    sim.draw_hoop(hoop, frame2, cam2)
    sim.draw_square(square, frame2, cam2)
    #draw_hoop(hoop2, frame1, cam1)

    S1 = -1 + cv2.getTrackbarPos("S1", "SJES")
    S2 = -1 + cv2.getTrackbarPos("S2", "SJES")
    S3 = -1 + cv2.getTrackbarPos("S3", "SJES")

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
        tvec = np.array([[-1,0,0],[0,1,0],[0,0,1]]).dot(Cvector) ### Correction for minus sign in the translation vector

        cv2.aruco.drawAxis(frame1, cam1.cameraMatrix, cam1.distCoeffs, rvec, tvec, 0.1)

    # cv2.putText(frame1, "position:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    # cv2.putText(frame1, "x={:.2f}".format(cam1.pos[0][0]), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    # cv2.putText(frame1, "y={:.2f}".format(cam1.pos[1][0]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    # cv2.putText(frame1, "z={:.2f}".format(cam1.pos[2][0]), (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    # cv2.putText(frame1, "orientation:", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    # cv2.putText(frame1, "yaw={:.2f}".format(cam1.yaw*180/np.pi), (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    # cv2.putText(frame1, "pitch={:.2f}".format(cam1.pitch*180/np.pi), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    # cv2.putText(frame1, "roll={:.2f}".format(cam1.roll*180/np.pi), (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    control = cv2.waitKeyEx(1)

    if control == 27:
        break
    if control == 102:
        fit_ellipse = not fit_ellipse
    if control == 119:              # w          FORWARD
        dcam.translate(0, 0.1, 0)
    if control == 115:              # s          BACKWARD
        dcam.translate(0, -0.1, 0)
    if control == 100:              # d          RIGHT
        dcam.translate(0.1, 0, 0)
    if control == 97:               # a          LEFT
        dcam.translate(-0.1, 0, 0)
    if control == 2490368:          # up arrow   UP
        dcam.translate(0, 0, 0.1)
    if control == 2621440:          # down arrow DOWN
        dcam.translate(0, 0, -0.1)
    if control == 2555904:          # right arrow
        dcam.rotate(0, 0, 0.01)
    if control == 2424832:          # left arrow
        dcam.rotate(0, 0, -0.01)

    vis = np.hstack((frame1, frame2))
    cv2.imshow("simulation", vis)

cv2.destroyAllWindows()

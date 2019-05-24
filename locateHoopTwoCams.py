import numpy as np
from numpy import linalg as LA
import itertools

def getEllipseParams(fit):
    (xc, yc), (a, b), theta = fit
    xc = xc-768
    yc = -yc+432
    a = a/2
    b = b/2
    theta = theta*np.pi/180

    A = a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2
    B = (b ** 2 - a ** 2) * np.sin(theta) * np.cos(theta)
    C = a ** 2 * np.cos(theta) ** 2 + b ** 2 * np.sin(theta) ** 2
    D = -A * xc - B * yc / 2
    E = -B * xc / 2 - C * yc
    F = A * xc ** 2 + B * xc * yc + C * yc ** 2 - a ** 2 * b ** 2

    return A, B, C, D, E, F


def estimate_pose(fit1, fit2):
    (x1, y1), _, _ = fit1
    (x2, y2), _, _ = fit2
    x1 = x1 - 768
    y1 = -y1 + 432
    x2 = x2 - 768
    y2 = -y2 + 432

    A, B, C, D, E, F = getEllipseParams(fit1)
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
    Cvector = centercoor(x1, y1, x2, cam1, cam2)

    Nvector = V * np.array([S2 * h, 0, -S1 * g])

    rvec, _ = cv2.Rodrigues(Rc)
    tvec = Cvector  ### Correction for minus sign in the translation vector
    return rvec, tvec


def centercoor(hoopx1, hoopy1, hoopx2, d, cam1, cam2):
    h1 = 1/cam1.cameraMatrix[0][0]
    h2 = 1/cam2.cameraMatrix[0][0]
    z = d/(hoopx1 * h1 - hoopx2 * h2)
    x = z * hoopx1 * h1
    y = z * hoopy1 * h2
    coorditates = np.array([x, y, z])
    return coorditates


def orientation(a1,b1,a2,b2):
    exc1 = b1/a1
    exc2 = b2/a2


def estimate_pose_2cams(fit1, fit2, d, cam1, cam2):
    (xc1, yc1), (ma1, MA1), theta1 = fit1
    (xc2, yc2), (ma2, MA2), theta2 = fit2
    xc1 = xc1 -480
    yc1 = -yc1 +360
    xc2 = xc2 -480
    yc2 = -yc2 +360
    tvec = centercoor(xc1, yc1, xc2, d, cam1, cam2)
    Tvec = np.dot(tvec, np.array([[1,0,0],[0,-1,0],[0,0,1]]))
    Rmatrix = np.identity(3)

    return Tvec, Rmatrix


def find_N(fit, cam):

    A, B, C, D, E, F = getEllipseParams(fit)
    print(A,B,C,D,E,F)
    f = cam.cameraMatrix[0][0]
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

    Nvector = np.dot(V, np.array([h, 0, g]))

    return Nvector
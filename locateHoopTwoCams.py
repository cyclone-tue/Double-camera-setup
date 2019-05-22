import numpy as np


def centercoor(hoopx1, hoopy1, hoopx2, cam1, cam2):
    d = cam2.pos[0]-cam1.pos[0]
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


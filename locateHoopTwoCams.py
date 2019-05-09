import numpy as np

h = 1/600  # meters (on screen 1m away) per pixel (in camera picture)
d = 0.1  # distance between cameras in meters


def centercoor(cam1x, cam1y, cam2x, cam2y):
    z = d/((cam1x-cam2x)*h)
    x = z * cam1x * h
    y = z* cam1y * h
    coorditates = np.array([[x],[y],[z]])
    return coorditates


def orientation(a1,b1,a2,b2):
    exc1 = b1/a1
    exc2 = b2/a2


print(centercoor(0.1,0,0.09,0))

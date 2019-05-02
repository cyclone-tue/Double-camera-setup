import numpy as np
import cv2

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


class Camera:
    def __init__(self, rMat, pos, cameraMatrix, distCoeffs):
        self.yaw = 0  # wrt world frame
        self.pitch = 0
        self.roll = 0
        self.rMat = rMat
        self.pos = pos  # wrt world frame
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
    grid = np.zeros([rows, cols, 3])
    for i in range(rows):
        for j in range(cols):
            grid[i][j] = [i * length - (rows - 1) * length / 2, j * length - (cols - 1) * length / 2, 0]
    return grid


def create_hoop(r, px, py, pz):
    hoop = np.zeros([41, 3])
    for i in range(41):
        hoop[i] = [px + r * np.cos(i * np.pi / 20), py, pz + r * np.sin(i * np.pi / 20)]
    return hoop

def create_square(r, px, py, pz):
    square = np.zeros([5,3])
    square[0] = [px - r, py, pz - r]
    square[1] = [px - r, py, pz + r]
    square[2] = [px + r, py, pz + r]
    square[3] = [px + r, py, pz - r]
    square[4] = [px - r, py, pz - r]
    return square


def project_grid(grid, cam):
    return np.array([
        cam.project(row)
        for row in grid])


def draw_grid(grid, img, cam):
    pgrid = project_grid(grid, cam)
    for i in range(pgrid.shape[0]):  # rows
        for j in range(pgrid.shape[1]):  # cols
            if i != 0:
                pt1 = tuple(pgrid[i][j][0])
                pt2 = tuple(pgrid[i - 1][j][0])
                cv2.line(img, pt1, pt2, (100, 100, 100), 1)
            if j != 0:
                pt1 = tuple(pgrid[i][j][0])
                pt2 = tuple(pgrid[i][j - 1][0])
                cv2.line(img, pt1, pt2, (100, 100, 100), 1)


def draw_hoop(hoop, img, cam):
    phoop = cam.project(hoop)
    for i in range(phoop.shape[0]):  # number of points
        if i != 0:
            pt1 = tuple(phoop[i][0])
            pt2 = tuple(phoop[i - 1][0])
            cv2.line(img, pt1, pt2, (0, 255, 0), 1)


def draw_square(square, img, cam):
    psquare = cam.project(square)
    for i in range(psquare.shape[0]):  # number of points
        if i != 0:
            pt1 = tuple(psquare[i][0])
            pt2 = tuple(psquare[i - 1][0])
            cv2.line(img, pt1, pt2, (0, 255, 0), 1)

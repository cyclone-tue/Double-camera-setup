import numpy as np
import cv2

dragging = False
xi, yi = -1, -1


# Calculates rotation matrix given euler angles.
def rotation_matrix(theta):
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
    def __init__(self, pos, theta, cameraMatrix, distCoeffs):
        # pose
        self.pos = pos
        self.theta = theta
        self.rMat = rotation_matrix(theta)

        # intrinsic camera parameters
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

    def translate(self, vector):
        self.pos += vector

    def rotate(self, theta):
        self.theta += theta
        self.rMat = rotation_matrix(self.theta)

    # projects 3d points from world frame to 2d camera image
    def project(self, points):
        # x-axis is used as projection axis
        M = np.dot(self.rMat, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))

        tvec = -np.dot(np.transpose(M), self.pos)
        rvec = cv2.Rodrigues(np.transpose(M))[0]
        return cv2.projectPoints(points, rvec, tvec, self.cameraMatrix, self.distCoeffs)[0].astype(np.int64)

    def mouse_control(self, event, x, y, flags, params):
        global xi, yi, dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            xi, yi = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                yaw = -np.pi * (x - xi) / 1536
                pitch = np.pi * (y - yi) / (2 * 864)
                self.rotate([0, pitch, yaw])
                xi, yi = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

    def key_control(self, key):
        if key == 119:      # w             FORWARD
            self.translate([0.1, 0, 0])
        if key == 115:      # s             BACKWARD
            self.translate([-0.1, 0, 0])
        if key == 100:      # d             RIGHT
            self.translate([0, 0.1, 0])
        if key == 97:       # a             LEFT
            self.translate([0, -0.1, 0])
        if key == 2490368:  # up arrow      UP
            self.translate([0, 0, -0.1])
        if key == 2621440:  # down arrow    DOWN
            self.translate([0, 0, 0.1])
        if key == 2555904:  # right arrow   ROLL CLOCKWISE
            self.rotate([0.1, 0, 0])
        if key == 2424832:  # left arrow    ROLL COUNTER CLOCKWISE
            self.rotate([-0.1, 0, 0])


class DoubleCamera(Camera):
    def __init__(self, cam1, cam2, pos):
        # cameras
        self.cam1 = cam1
        self.cam2 = cam2
        # pose
        self.pos = pos
        self.theta = np.zeros(3)
        self.rMat = np.identity(3)

    def translate(self, vector):
        self.pos += vector
        self.cam1.translate(vector)
        self.cam2.translate(vector)

    def rotate(self, theta):
        self.cam1.rotate(theta)
        self.cam2.rotate(theta)

        # update postions
        self.cam1.pos = self.pos + np.dot(np.transpose(self.rMat), (self.cam1.pos - self.pos))
        self.cam2.pos = self.pos + np.dot(np.transpose(self.rMat), (self.cam2.pos - self.pos))

        self.theta += theta
        self.rMat = rotation_matrix(self.theta)

        self.cam1.pos = self.pos + np.dot(self.rMat, (self.cam1.pos - self.pos))
        self.cam2.pos = self.pos + np.dot(self.rMat, (self.cam2.pos - self.pos))


class Mesh:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.pos = np.array([0., 0., 0.])
        self.theta = np.array([0., 0., 0.])

    def draw(self, img, cam, color=(100, 100, 100), pt=1):
        pvertices = cam.project(self.vertices)
        for edge in self.edges:
            pt1 = tuple(pvertices[edge[0]][0])
            pt2 = tuple(pvertices[edge[1]][0])
            cv2.line(img, pt1, pt2, color, pt)

    def translate(self, vector):
        self.pos += vector
        for vertex in self.vertices:
            vertex += vector

    def rotate(self, theta):
        M1 = np.transpose(rotation_matrix(self.theta))
        M2 = rotation_matrix(theta)
        R = np.dot(M2, M1)
        for vertex in self.vertices:
            delta = self.pos + np.dot(R, vertex - self.pos) - vertex
            vertex += delta
        self.theta = theta


def create_grid(rows, cols, length):
    vertices = np.zeros([rows * cols, 3])
    edges = []
    for i in range(rows):
        for j in range(cols):
            vertices[i * cols + j] = [
                i * length - (rows - 1) * length / 2,
                j * length - (cols - 1) * length / 2,
                0.
            ]
            if i != 0:
                edges.append((cols * (i - 1) + j, cols * i + j))
            if j != 0:
                edges.append((cols * i + j - 1, cols * i + j))
    return Mesh(vertices, edges)


def create_path(vertices, loop=False):
    edges = [(i, i+1) for i in range(len(vertices)-1)]
    if loop:
        edges.append((0, len(vertices)-1))
    return Mesh(vertices, edges)


def create_hoop(r, px, py, pz, num=40):
    vertices = np.array([[
        px,
        py + r * np.cos(i * 2 * np.pi / num),
        pz + r * np.sin(i * 2 * np.pi / num)
    ] for i in range(num)])
    return create_path(vertices, loop=True)


def create_square(r, px, py, pz):
    vertices = np.zeros([5,3])
    vertices[0] = [px, py - r, pz - r]
    vertices[1] = [px, py - r, pz + r]
    vertices[2] = [px, py + r, pz + r]
    vertices[3] = [px, py + r, pz - r]
    vertices[4] = [px, py - r, pz - r]
    return create_path(vertices, loop=True)


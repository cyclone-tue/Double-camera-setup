import numpy as np
import cv2
import simulation as sim
from locateHoopTwoCams import estimate_pose_2cams, find_N


cam1 = sim.Camera(
    pos=np.array([-5., -.1, -2.]),
    theta=np.zeros(3),
    cameraMatrix=np.array([[6.e+02, 0., 1.5*320.], [0., 6.e+02, 1.5*240.], [0., 0., 1.]]),
    distCoeffs=np.array([0., 0., 0., 0., 0.])
    )


cam2 = sim.Camera(
    pos=np.array([-5., .1, -2.]),
    theta=np.zeros(3),
    cameraMatrix=np.array([[6.e+02, 0., 1.5*320.], [0., 6.e+02, 1.5*240.], [0., 0., 1.]]),
    distCoeffs=np.array([0., 0., 0., 0., 0.])
    )

dcam = sim.DoubleCamera(
    cam1=cam1,
    cam2=cam2,
    pos=np.array([-5., 0., -2.])
)


# creating graphics
grid = sim.create_grid(10, 10, 0.3)
hoop = sim.create_hoop(1, px=0, py=0, pz=-2)
square = sim.create_square(1, px=0, py=0, pz=-2)


cv2.namedWindow('simulation')
cv2.setMouseCallback('simulation', dcam.mouse_control)

fit_ellipse = False


while True:
    # using screen resolution of 1536x864
    frame1 = np.zeros((720, 960, 3), dtype=np.uint8)  # cv2.imread("images.jpg")
    frame2 = np.zeros((720, 960, 3), dtype=np.uint8)  # shape = (480, 640, 3)

    grid.draw(frame1, cam1)
    hoop.draw(frame1, cam1, color=(0, 255, 0), pt=1)
    square.draw(frame1, cam1, color=(0, 255, 0), pt=1)

    grid.draw(frame2, cam2)
    hoop.draw(frame2, cam2, color=(0, 255, 0), pt=1)
    square.draw(frame2, cam2, color=(0, 255, 0), pt=1)

    cv2.rectangle(frame1, (0, 0), (960, 720), (255, 255, 255), 1)
    cv2.rectangle(frame2, (0, 0), (960, 720), (255, 255, 255), 1)

    hoop1 = cam1.project(hoop.vertices)
    hoop2 = cam2.project(hoop.vertices)

    if fit_ellipse and len(hoop1) > 10 and len(hoop2) > 10:
        fit1 = cv2.fitEllipse(hoop1)
        fit2 = cv2.fitEllipse(hoop2)

        cv2.ellipse(frame1, fit1, (255, 0, 0), 5)
        cv2.ellipse(frame2, fit2, (255, 0, 0), 5)

        # draw ellipse long and short axis  (RED = LONG AXIS) (BLUE = SHORT AXIS)
        (xc, yc), (ma, MA), theta = fit1
        theta = theta * np.pi / 180
        a, b = MA/2, ma/2
        cv2.circle(frame1, (int(xc), int(yc)), 2, (255, 255, 255), 1)
        cv2.line(frame1, (int(xc), int(yc)), (int(xc - a * np.sin(theta)), int(yc + a * np.cos(theta))),
                 (0, 0, 255), 1)
        cv2.line(frame1, (int(xc), int(yc)), (int(xc + b * np.cos(theta)), int(yc + b * np.sin(theta))),
                 (255, 0, 0), 1)

        # draw ellipse long and short axis  (RED = LONG AXIS) (BLUE = SHORT AXIS)
        (xc, yc), (ma, MA), theta = fit2
        theta = theta * np.pi / 180
        a, b = MA/2, ma/2
        cv2.circle(frame2, (int(xc), int(yc)), 2, (255, 255, 255), 1)
        cv2.line(frame2, (int(xc), int(yc)), (int(xc - a * np.sin(theta)), int(yc + a * np.cos(theta))),
                 (0, 0, 255), 1)
        cv2.line(frame2, (int(xc), int(yc)), (int(xc + b * np.cos(theta)), int(yc + b * np.sin(theta))),
                 (255, 0, 0), 1)

        translation, Rotation = estimate_pose_2cams(fit1, fit2, 0.2, cam1, cam2)


        rvec, _ = cv2.Rodrigues(Rotation)

        #cv2.aruco.drawAxis(frame1, cam1.cameraMatrix, cam1.distCoeffs, rvec, translation, 0.1)
        translationcam2 = translation - np.array([0.2,0,0])
        cv2.aruco.drawAxis(frame2, cam2.cameraMatrix, cam2.distCoeffs, rvec, translationcam2, 0.1)
        Nvec = find_N(fit1, cam1)
        print(translation, translation+Nvec)
        Nvec_projection=sim.create_Nvec(translation,Nvec)
        Nvec_projection.draw(frame1, cam1, color=(0, 255, 0), pt=5)

    key = cv2.waitKeyEx(1)
    dcam.key_control(key)

    if key == 102:
        fit_ellipse = not fit_ellipse

    if key == 27:
        break

    vis = np.hstack((frame1, frame2))
    cv2.imshow("simulation", vis)

cv2.destroyAllWindows()

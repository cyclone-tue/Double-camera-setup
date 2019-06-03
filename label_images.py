import numpy as np
import cv2
import os


x, y = 0, 0


def set_box(event, xi, yi, flags, param):
    global x, y
    x, y = xi, yi
    if event == cv2.EVENT_LBUTTONUP:
        print('x=', xi, file=label)
        print('y=',yi, file=label)


cv2.namedWindow('image')

counter = 0
for file in os.listdir(os.fsencode('Dataset')):
    filename = os.fsdecode(file)
    counter += 1
    while True:
        frame = cv2.imread('Dataset/' + filename)
        label = open('label' + str(counter) + '.txt', 'w')
        cv2.setMouseCallback('image', set_box)
        # x axis
        cv2.line(frame, (0, y), (frame.shape[1], y), (255, 0, 0), 1)
        # y axis
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 0, 0), 1)

        cv2.imshow('image', frame)
        key = cv2.waitKey(1)
        if key == 27:
            label.close()
            break
        if key == 32:
            break
    if key == 27:
        break

cv2.destroyAllWindows()
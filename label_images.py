import numpy as np
import cv2
import os


x, y = 0, 0
corner1 = (x, y)
corner2 = (x, y)
drawing = False


def set_box(event, xi, yi, flags, param):
    global x, y, corner1, corner2, drawing
    x, y = xi, yi
    if event == cv2.EVENT_LBUTTONUP:
        if not drawing:
            corner1 = (x, y)
            corner2 = (x, y)
        drawing = not drawing
    if drawing:
        corner2 = (x, y)


cv2.namedWindow('image')
cv2.setMouseCallback('image', set_box)

dataset = os.listdir(os.fsencode('Dataset'))
labels = os.listdir(os.fsencode('Labels'))

# create empty (zeros) label files
# for index, file in enumerate(dataset):
#     box = np.zeros(4, dtype=np.int16)
#     np.savetxt('Labels/box' + str(index + 1) + '.txt', box, fmt='%i')

data_pairs = list(zip(dataset, labels))
index = 0

while True:
    image_file, label_file = data_pairs[index]
    image_filename = os.fsdecode(image_file)
    label_filename = os.fsdecode(label_file)

    # set title
    cv2.setWindowTitle('image', image_filename)

    # load label
    if not drawing:
        box = np.loadtxt('Labels/' + label_filename, dtype=np.int16)
        corner1 = tuple(box[0:2])
        corner2 = tuple(box[2:4])

    while True:
        key = cv2.waitKeyEx(1)

        image = cv2.imread('Dataset/' + image_filename)
        # x axis
        cv2.line(image, (0, y), (image.shape[1], y), (255, 0, 0), 1)
        # y axis
        cv2.line(image, (x, 0), (x, image.shape[0]), (255, 0, 0), 1)
        # rectangle
        cv2.rectangle(image, corner1, corner2, (255, 0, 0), 1)
        # transparent overlay
        overlay = image.copy()
        cv2.rectangle(overlay, corner1, corner2, (0, 0, 0, 0.4), cv2.FILLED)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.imshow('image', image)
        if key != -1:
            break

    if key == 2555904 or key == 32:     # right arrow or space bar
        if not drawing:
            box = corner1 + corner2
            np.savetxt('Labels/' + label_filename, box, fmt='%i')
            print('saved ' + label_filename)
        index += 1
        drawing = False
    if key == 2424832 and index > 0:    # left arrow
        index -= 1
        drawing = False
    if key == 27:                       # escape
        break

cv2.destroyAllWindows()


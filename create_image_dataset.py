import numpy as np
import cv2

cap = cv2.VideoCapture(0)
counter = 348

while True:
    _, frame = cap.read()

    cv2.imshow('image', frame)
    key = cv2.waitKey(1)
    if key == 32:
        counter += 1
        cv2.imwrite('Dataset/image' + str(counter) + '.png', frame)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2


def find_contours(gray, labels, markers):
    black = np.zeros(gray.shape, np.uint8)

    n = 0
    Areas = []
    contours = []

    for label in np.unique(labels):
        if label == 0:
            continue
        # Create a mask
        mask = np.zeros(markers.shape, dtype="uint8")
        mask[labels == label] = 255
        # Find contours and determine contour area
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        contours.append(c)
        area = cv2.contourArea(c)
        if area > 10:
            Areas.append(area)
        n += 1
        cv2.drawContours(black, [c], -1, (255, 255, 255), -1)
        cv2.drawContours(black, [c], -1, (0, 0, 0), 2)
        cv2.drawContours(gray, [c], -1, (0, 0, 0), 1)
    cnts = cv2.findContours(markers.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(black, cnts, -1, (255, 0, 0), -1)
    return gray, black
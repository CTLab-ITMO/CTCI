import cv2
import numpy as np
import sys
import os


def process_image(img: np.array):
    _, res = cv2.threshold(
        cv2.morphologyEx(
            cv2.equalizeHist(img),
            cv2.MORPH_BLACKHAT,
            kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
        ),
        40, 255,
        cv2.THRESH_BINARY_INV,
    )

    res2 = cv2.erode(
        cv2.morphologyEx(
            res,
            cv2.MORPH_CLOSE,
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
        ),
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
        iterations=1
        )
    return res2


if __name__=="__main__":
    folder = sys.argv[1]
    result_folder = sys.argv[2]

    for im_name in os.listdir(folder):
        img = cv2.imread(
            os.path.join(folder, im_name), 0
        )
        result = process_image(img)
        mask_name = im_name[:-4] + "_mask" + im_name[-4:]
        cv2.imwrite(
            os.path.join(result_folder, mask_name),
            result
        )



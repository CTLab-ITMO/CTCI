import cv2
import os
import numpy as np

folder = rf'D:\vscode\ctci\CTCI\data\real_covdor_masks'

names = [
    '2024-12-07T20_25_59',
    '2024-12-07T20_26_59',
    '2024-12-07T20_27_59',
    '2024-12-07T22_36_54',
    '2024-12-07T22_37_54',
    '2024-12-07T22_38_54',
    '2024-12-07T22_39_54',
    '2024-12-07T22_40_54',
    '2024-12-07T22_41_58',
    '2024-12-07T22_42_58',
    '2024-12-07T22_43_58',
    '2024-12-07T22_44_58',
    '2024-12-07T23_05_17',
    '2024-12-07T23_06_17',
    '2024-12-07T23_07_20',
    '2024-12-08T02_47_08',
    '2024-12-08T17_13_22',
    '2024-12-08T17_16_25',
    '2024-12-08T17_47_54',
    '2024-12-08T17_52_59',
]

for imfile in os.listdir(folder):
    for n in names:
        if imfile.startswith(n):
            img = cv2.imread(os.path.join(folder, imfile), 0)
            cv2.imwrite(os.path.join(folder, imfile), np.zeros_like(img))


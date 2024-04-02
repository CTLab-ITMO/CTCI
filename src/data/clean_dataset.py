import os
import os.path as osp
import sys

import cv2
import numpy as np
from tqdm import tqdm
from natsort import natsorted


def clean_dataset(path):

    images_path = osp.join(path, "images")
    masks_path = osp.join(path, "masks")

    images_list = natsorted(os.listdir(images_path), key=str)

    prev = {
        'image_path': images_list[0],
        'image': cv2.imread(osp.join(images_path, images_list[0]))
    }

    for image_path in tqdm(images_list[1:]):
        im = cv2.imread(osp.join(images_path, image_path))

        if np.array_equal(im, prev['image']):

            os.remove(osp.join(images_path, prev['image_path']))
            os.remove(osp.join(masks_path, prev['image_path']))
            print("deleted {}".format(image_path))

        prev['image'] = im
        prev['image_path'] = image_path


if __name__ == '__main__':
    path = sys.argv[1]
    clean_dataset(path)
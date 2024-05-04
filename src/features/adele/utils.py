import os
import os.path as osp

import cv2

artifact_path = r'./data/temp/_label_corrections/'


def create_labels_artifact():
    if not osp.exists(artifact_path):
        os.mkdir(artifact_path)


def convert_data_to_dict(filenames, labels):
    data = {}
    for name, label in zip(filenames, labels):
        data[name] = label.cpu().numpy().transpose(1, 2, 0) * 255
    return data


def write_labels(data: dict, path=artifact_path):
    for name, label in data.items():
        cv2.imwrite(
            osp.join(path, name),
            label
        )


def read_label(filename, path=artifact_path):
    return cv2.imread(
        osp.join(path, filename),
        0
    )

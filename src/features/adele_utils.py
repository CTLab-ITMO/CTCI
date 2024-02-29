import os.path as osp
import json


artifact_path = '../data/temp/_label_corrections.json'


def create_labels_artifact(path=artifact_path):
    if not osp.exists(path):
        with open(artifact_path, "w") as f:
            json.dump({}, f)


def convert_data_to_dict(filenames, labels):
    data = {}
    for name, label in zip(filenames, labels):
        data[name] = label
    return data


def write_labels(data: dict, path = artifact_path):
     with open(path, "w") as f:
         json.dump(data, f)


def read_label(filename, path=artifact_path):
    with open(path, "r") as f:
        data = json.load(f)
        label = data[filename]
    return label
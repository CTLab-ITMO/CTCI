import os

import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch

from src.models.metrics import TemporalConsistency
from src.models.metrics import OpticalFlowSimilarity, ObjectsRecall
from src.models.utils.reproducibility import set_seed


def video_metrics_report(images_dir, masks_dir):
    set_seed(seed=0)

    image_list = os.listdir(images_dir)
    masks_list = os.listdir(masks_dir)

    temporal_consistency = TemporalConsistency()
    optical_flow_sim = OpticalFlowSimilarity()
    obj_recall = ObjectsRecall()

    temporal_consistency_list = []
    optical_flow_sim_list = []
    objects_recall_list = []

    for image_name, mask_name in zip(image_list, masks_list):
        image_path = os.path.join(images_dir, image_name)
        mask_path = os.path.join(masks_dir, mask_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # temporal consistency
        if len(objects_recall_list) != 0:
            pass

        # optical_flow_sim
        if len(objects_recall_list) != 0:
            pass

        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        obj_recall_num = obj_recall(image, mask)
        objects_recall_list.append(obj_recall_num)

    # calc average metrics


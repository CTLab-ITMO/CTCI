import os

import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch

from src.models.metrics import TemporalConsistency
from src.models.metrics import OpticalFlowSimilarity, ObjectsRecall
from src.models.utils.reproducibility import set_seed


def get_unmoving_mask(prev_frame: np.array, cur_frame: np.array):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_RGB2GRAY)

    # Compute absolute difference between frames
    diff = cv2.absdiff(prev_gray, cur_gray)

    # Apply threshold to get binary mask where differences are significant
    _, mask = cv2.threshold(diff, 3, 255, cv2.THRESH_BINARY)
    return mask / 255

def video_metrics_report(images_dir, masks_dir):
    set_seed(seed=0)

    image_list = os.listdir(images_dir)
    masks_list = os.listdir(masks_dir)

    device = 'cuda'
    temporal_consistency = TemporalConsistency(device=device)
    optical_flow_sim = OpticalFlowSimilarity(device=device)
    obj_recall = ObjectsRecall()

    temporal_consistency_list = []
    optical_flow_sim_list = []
    objects_recall_list = []

    prev_image = None
    prev_mask = None

    for i, (image_name, mask_name) in enumerate(zip(image_list, masks_list)):
        if i < 1:
            continue
        if i >= 150:
            break

        image_path = os.path.join(images_dir, image_name)
        mask_path = os.path.join(masks_dir, mask_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # temporal consistency
        if not (prev_image is None):
            prev_image_tensor = torch.tensor(prev_image / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device)
            image_tensor = torch.tensor(image / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device)

            prev_mask_tensor = torch.tensor(prev_mask / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device)
            mask_tensor = torch.tensor(mask / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device)

            temporal_consistency_num = temporal_consistency(
                prev_image_tensor, image_tensor,
                prev_mask_tensor, mask_tensor
            )
            temporal_consistency_list.append(temporal_consistency_num.item())

        # optical_flow_sim
        if not (prev_image is None):
            prev_image_tensor = torch.tensor(prev_image / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
            image_tensor = torch.tensor(image / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)

            prev_mask_tensor = torch.tensor(prev_mask / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
            mask_tensor = torch.tensor(mask / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)

            optical_flow_sim_num = optical_flow_sim(
                prev_image_tensor, image_tensor,
                prev_mask_tensor, mask_tensor
            )
            optical_flow_sim_list.append(optical_flow_sim_num.item())

        if not (prev_image is None):
            unmoving_mask = get_unmoving_mask(prev_image, image)
            obj_recall_num = obj_recall(image, mask, unmoving_mask=unmoving_mask)
            objects_recall_list.append(obj_recall_num)

        prev_image = image
        prev_mask = mask

    # calc average metrics
    temporal_consistency_metric = sum(temporal_consistency_list) / len(temporal_consistency_list)
    optical_flow_sim_metric = sum(optical_flow_sim_list) / len(optical_flow_sim_list)
    objects_recall_metric = sum(objects_recall_list) / len(objects_recall_list)

    return temporal_consistency_metric, optical_flow_sim_metric, objects_recall_metric


if __name__ == "__main__":
    images_dir = r"C:\Internship\ITMO_ML\CTCI\data\test_data\metrics\images"
    masks_dir = r"C:\Internship\ITMO_ML\CTCI\data\test_data\metrics\masks"
    labels_dir = r"C:\Internship\ITMO_ML\CTCI\data\test_data\metrics\label"

    temporal_consistency_metric, optical_flow_sim_metric, objects_recall_metric = video_metrics_report(images_dir, masks_dir)
    print("OURS: ")
    print(f" temporal_consistency: {temporal_consistency_metric}")
    print(f" optical_flow_similarity: {optical_flow_sim_metric}")
    print(f" objects_recall: {objects_recall_metric}")
    print("-------------\n\n")

    temporal_consistency_metric, optical_flow_sim_metric, objects_recall_metric = video_metrics_report(images_dir, labels_dir)
    print("WATERSHED:")
    print(f" temporal_consistency: {temporal_consistency_metric}")
    print(f" optical_flow_similarity: {optical_flow_sim_metric}")
    print(f" objects_recall: {objects_recall_metric}")
    print("-------------\n\n")
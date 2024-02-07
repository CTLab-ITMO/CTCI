import os

import torch

from src.sam_segmentation.yolo import load_yolov8_detector
from src.sam_segmentation.sam import load_sam_predictor, segment_images_from_folder


if __name__ == "__main__":
    data_dir = r"C:\Internship\ITMO_ML\data\raw_frames\Frames_bubbles\Bubbles_split"
    folder = r"train"

    custom_yolo_checkpoint_path = r"C:\Internship\ITMO_ML\CTCI\checkpoints\yolov8\yolov8s_on_mid_bubbles_69\run2_80_epochs\weights\best.pt"
    sam_checkpoint = r"C:\Internship\ITMO_ML\CTCI\checkpoints\sam_checkpoints\sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    target_length = 1024
    narrowing = 0.20
    erode_iterations = 1
    processes_num = 0

    detector = load_yolov8_detector(custom_yolo_checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = load_sam_predictor(checkpoint_path=sam_checkpoint, model_type=model_type, device=device)

    source_dir = os.path.join(data_dir, folder)
    output_dir = os.path.join(data_dir, folder + "_masks")

    segment_images_from_folder(
        source_dir,
        output_dir,
        detector, predictor,
        target_length=target_length,
        narrowing=narrowing,
        erode_iterations=erode_iterations,
        processes_num=processes_num
    )


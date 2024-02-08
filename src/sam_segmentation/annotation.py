import os
import sys

import torch

from src.sam_segmentation.yolo import load_yolov8_detector
from src.sam_segmentation.sam import load_sam_predictor, segment_images_from_folder


def annotation(
        data_dir: str, folder: str,
        custom_yolo_checkpoint_path: str, sam_checkpoint: str, sam_model_type: str,
        target_length: int = 1024, narrowing: float = 0.20, erode_iterations: int = 1, processes_num: int = 0,
        device: str = "cpu"
):
    if sam_model_type not in ["vit_b", "vit_l", "vit_h"]:
        print("Undefined sam model type")
        # TODO: Throw error
        return

    detector = load_yolov8_detector(custom_yolo_checkpoint_path)

    if device != "cpu":
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True

    predictor = load_sam_predictor(checkpoint_path=sam_checkpoint, model_type=sam_model_type, device=device)

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


if __name__ == "__main__":
    # TODO: add argparser

    data_dir = sys.argv[1]
    folder = sys.argv[2]

    custom_yolo_checkpoint_path = sys.argv[3]
    sam_checkpoint = sys.argv[4]
    sam_model_type = sys.argv[5]

    target_length = 1024
    narrowing = 0.20
    erode_iterations = 1
    processes_num = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    annotation(
        data_dir, folder,
        custom_yolo_checkpoint_path, sam_checkpoint, sam_model_type,
        target_length=target_length,  narrowing=narrowing,
        erode_iterations=erode_iterations, processes_num=processes_num,
        device=device
    )

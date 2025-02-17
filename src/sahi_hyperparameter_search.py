from sahi.predict import get_sliced_prediction

import cv2
from src.utils.masks import masks_narrowing, unite_masks
from src.annotation.yolo import load_yolo_sahi_detector
from src.annotation.sam import sam_segmentation, load_sam_predictor
import numpy as np
import itertools


def yolo_sahi_detect(
        image,
        detector,
        shape_scale=2,
        slice_scale=4,
        overlap_ratio=0.1,
        postprocess_type='NMS'
):
    original_h, original_w = image.shape[:2]
    resized_h, resized_w = original_h * shape_scale, original_w * shape_scale

    image_resized = cv2.resize(image, (resized_w, resized_h))

    result = get_sliced_prediction(
        image_resized,
        detector,
        slice_height=resized_h // slice_scale,
        slice_width=resized_w // slice_scale,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type=postprocess_type,
    )

    object_prediction_list = result.object_prediction_list
    boxes = []

    scale_x = original_w / resized_w
    scale_y = original_h / resized_h

    for object_prediction in object_prediction_list:
        x1, y1, x2, y2 = object_prediction.bbox.to_xyxy()

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        boxes.append([x1, y1, x2, y2])

    return boxes


if __name__ == "__main__":
    image = cv2.imread("./data/covdor/192.168.1.11_2024-12-07T17:58:07.png")
    detector = load_yolo_sahi_detector("./models/annotation/11rocks.pt")
    predictor = load_sam_predictor("./models/annotation/sam_vit_b_01ec64.pth", model_type="vit_b")

    # Гиперпараметры для грид-серча
    shape_scales = [1, 2, 3]
    slice_scales = [2, 4, 6]
    overlap_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    postprocess_types = ['NMM', 'GREEDYNMM', 'NMS']

    param_combinations = list(itertools.product(shape_scales, slice_scales, overlap_ratios, postprocess_types))

    for idx, param_combination in enumerate(param_combinations):
        boxes = yolo_sahi_detect(
            image=image,
            detector=detector,
            shape_scale=param_combination[0],
            slice_scale=param_combination[1],
            overlap_ratio=param_combination[2],
            postprocess_type=param_combination[3],
        )

        masks_list = sam_segmentation(
            image=image, predictor=predictor, boxes=boxes, prompt_points=False, target_length=1024
        )
        masks_united = [unite_masks(masks) for masks in masks_list]
        masks_narrowed = masks_narrowing(masks_united, narrowing=0.2)
        mask_sam = unite_masks(masks_narrowed)

        mask = mask_sam[:, :, np.newaxis].repeat(3, axis=2)

        alpha = 0.6
        vis = image * alpha + (1 - alpha) * mask
        vis = vis.astype(np.uint8)

        cv2.imwrite(
            f"./data/exp/sh_{param_combination[0]},sl_{param_combination[1]},op_{param_combination[2]},p_ {param_combination[3]}.png",
            vis)

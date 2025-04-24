from sahi.predict import get_sliced_prediction

import cv2
import numpy as np
import itertools
from sahi import AutoDetectionModel
from src.utils.masks import masks_narrowing, unite_masks, suppress_watershed_with_yolosam
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import torch


def load_yolo_sahi_detector(checkpoint_path: str):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=checkpoint_path,
        confidence_threshold=0.3,
        device="cpu",  # or 'cuda:0'
    )
    return detection_model


def load_sam_predictor(checkpoint_path: str, model_type: str, device: str = "cpu") -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint_path)
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    print(f"Predictor's device is {predictor.device}")

    return predictor


def sam_segmentation(
        image, predictor: SamPredictor,
        boxes: list, prompt_points: bool = True, target_length: int = 1024
):

    device = predictor.device

    original_image_size = (image.shape[0], image.shape[1])
    transform = ResizeLongestSide(target_length=target_length)

    transformed_image = transform.apply_image(image)
    transformed_image_torch = torch.as_tensor(transformed_image, device=device)
    transformed_image_torch = transformed_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    predictor.set_torch_image(
        transformed_image=transformed_image_torch,
        original_image_size=original_image_size
    )

    boxes = np.array(boxes)
    boxes = transform.apply_boxes(boxes, (predictor.original_size[0], predictor.original_size[1]))
    boxes_tensor = torch.Tensor(boxes).float().to(device)

    if prompt_points:
        points = [np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]) for box in boxes]
        points = np.array(points)
        points = transform.apply_coords(points, (predictor.original_size[0], predictor.original_size[1]))
        points_tensor = torch.Tensor(points).float().to(device)
        labels_tensor = torch.ones((points.shape[0], points.shape[1])).float().to(device)
        masks_list, _, _ = predictor.predict_torch(
            boxes=boxes_tensor,
            point_coords=points_tensor,
            point_labels=labels_tensor,
            multimask_output=False
        )

    else:
        masks_list, _, _ = predictor.predict_torch(
            boxes=boxes_tensor,
            multimask_output=False,
            point_coords=None,
            point_labels=None
        )

    masks_list = masks_list.float().to("cpu").numpy()
    return masks_list


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
    print(torch.cuda.is_available())
    image = cv2.imread("./data/covdor/192.168.1.11_2024-12-07T16_42_55.png")
    detector = load_yolo_sahi_detector("./models/annotation/11rocks.pt")
    predictor = load_sam_predictor("./models/annotation/sam_vit_h.pth", model_type="vit_h", device="cuda")

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

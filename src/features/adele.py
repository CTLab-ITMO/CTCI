"""
adele implementation for c=1
"""

import torch
import torch.nn.functional as F


def _count_iou_slope(first, last):
    """
    this function implements calculating the slope of IoU metric
    while training segmentation models.
    its value should be thresholded to decide whether to use adele or not.


    the slope is calculated as given below:
    |f'(1) - f'(t)| / |f'(1)|

    Args:
        first (torch.float): first calculated iou
        last (torch.float): last calculated iou
    Returns:
        torch.float: slope value
    """
    slope = torch.divide(torch.abs(first - last), torch.abs(last))
    return slope


def if_update(first, last, thresh=0.9):
    """_summary_

    Args:
        first : first iou value
        last : last observed iou value
        thresh : slope threshold. Defaults to 0.9.

    Returns:
        bool: if masks should be updated
    """
    return _count_iou_slope(first=first, last=last) > thresh


def _interpolate_img(image, scale):
    """
    wrapper for interpolating image

    Args:
        image : input images
        scale : scale value

    Returns:
        torch.tensor: interpolated image
    """
    _, _, h, w = image.shape()
    return F.interpolate(image, [h*scale, w*scale], mode='bilinear', align_corners=True)


def _predict_on_scale(model, image, scale):
    """
    return model output on interpolated image and then rescale it back

    Args:
        model : model to predict
        image : images for prediction
        scale : scale value

    Returns:
        torch.tensor: predicted output
    """
    scaled = _interpolate_img(image, scale)
    res = model(scaled)
    out = _interpolate_img(res, 1/scale)
    return out


def predict_average_on_scales(model, batch, scales):
    """
    predict on scaled images, rescale and average over batch

    Args:
        model : model for prediction
        batch : batch of images
        scales (list): list of scales

    Returns:
        torch.tensor: averaged output
    """
    preds = []
    for s in scales:
        preds.append(_predict_on_scale(model, batch, s))
    preds = torch.tensor(preds, dtype=torch.float)
    return torch.mean(preds, dim=0).squeeze(0)


def correct_mask(batch, target, average, confidence_thresh=0.8):
    """
    mask correction method

    Args:
        batch : batch size
        target : batch of target tensors
        average : tensor of averaged output from scaled prediction
        confidence_thresh (float, optional): average confidence threshold. Defaults to 0.8.
    Returns:
        torch.tensor: corrected labels
    """
    indices = average >= confidence_thresh
    label = torch.argmax(average[indices])
    new_target = []
    for b in batch:
       new_target.appned(
           torch.where(indices, target[b], label)
       )
    new_target = torch.tensor(new_target)
    return new_target
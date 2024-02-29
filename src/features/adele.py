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


def _interpolate_img(image, scale=None, size=None):
    """
    wrapper for interpolating image

    Args:
        image : input images
        scale : scale value
        size : tuple to resize to

    Returns:
        torch.tensor: interpolated image
    """
    if scale:
        return F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True)
    return F.interpolate(image, size=size, mode='bilinear', align_corners=True)


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
    _, _, h, w = image.size()
    scaled = _interpolate_img(image, scale)
    res = model.predict(scaled)
    out = _interpolate_img(res, size=(h,w))
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
    preds = torch.stack(preds)
    return torch.mean(preds, dim=0).squeeze(0)


def correct_mask(target, average, confidence_thresh=0.8):
    """
    mask correction method

    Args:
        target : batch of target tensors
        average : tensor of averaged output from scaled prediction
        confidence_thresh (float, optional): average confidence threshold. Defaults to 0.8.
    Returns:
        torch.tensor: corrected labels
    """
    indices = average >= confidence_thresh
    new_target = []
    for t, i, a in zip(target, indices, average):
       masked = a * i.int().float()
       label = torch.argmax(masked, dim=0, keepdim=True).float()
       new_target.append(
           torch.where(i, t, label)
       )
    new_target = torch.stack(new_target)
    return new_target
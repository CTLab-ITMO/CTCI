"""
adele implementation for c=1
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


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
    wrapper for interpolating images

    Args:
        image : input images
        scale : scale value
        size : tuple to resize to

    Returns:
        torch.tensor: interpolated images
    """
    if scale:
        size = tuple(map(lambda x: int(x*scale), size))
        return TF.resize(image, size=size)
    return TF.resize(image, size=size)


def _predict_on_scale(model, image, scale):
    """
    return model output on interpolated images and then rescale it back

    Args:
        model : model to predict
        image : images for prediction
        scale : scale value

    Returns:
        torch.tensor: predicted output
    """
    _, _, h, w = image.size()
    scaled = _interpolate_img(image, scale=scale, size=(h,w))
    res = model.predict(scaled).float()
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


def correct_mask(target, average, confidence_thresh=0.6):
    """
    mask correction method

    Args:
        target : batch of target tensors
        average : tensor of averaged output from scaled prediction
        confidence_thresh (float, optional): average confidence threshold. Defaults to 0.6.
    Returns:
        torch.tensor: corrected labels
    """
    new_target = []
    for t in target:
        new_target.append(torch.where(average > confidence_thresh, 1, t))
    new_target = torch.stack(new_target)
    return new_target
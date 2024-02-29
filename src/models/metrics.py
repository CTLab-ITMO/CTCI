import torch
import torch.nn as nn


class IoUMetric(nn.Module):
    """
    Metric for binary segmentation

    Class of Intersection over Union metric

    It is calculated as the ratio between the overlap of
    the positive instances between two sets, and their mutual combined values

    J(A, B) = |A and B| / |A or B| = |A and B| / (|A| + |B| - |A and B|)

    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)

        return iou


class DiceMetric(nn.Module):
    """
    Metric for binary segmentation
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-5):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        sum = inputs.sum() + targets.sum()
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + eps) / (sum + eps)
        return dice


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-5):
        tp = torch.sum(inputs * targets)
        fp = torch.sum(inputs * (1 - targets))
        fn = torch.sum((1 - inputs) * targets)
        tn = torch.sum((1 - inputs) * (1 - targets))

        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        return accuracy


class Precision(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-5):
        tp = torch.sum(inputs * targets)
        fp = torch.sum(inputs * (1 - targets))

        precision = (tp + eps) / (tp + fp + eps)
        return precision


class Recall(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-5):
        tp = torch.sum(inputs * targets)
        fn = torch.sum((1 - inputs) * targets)

        recall = (tp + eps) / (tp + fn + eps)

        return recall

class MAP(nn.Module):
    """
    Metric for binary segmentation
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        pass
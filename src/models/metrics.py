import torch.nn as nn

import torch
import numpy as np
from tqdm import tqdm


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

    def __str__(self):
        return "iou"


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

    def __str__(self):
        return "dice"


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

    def __str__(self):
        return "accuracy"


class Precision(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-5):
        tp = torch.sum(inputs * targets)
        fp = torch.sum(inputs * (1 - targets))

        precision = (tp + eps) / (tp + fp + eps)
        return precision

    def __str__(self):
        return "precision"


class Recall(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-5):
        tp = torch.sum(inputs * targets)
        fn = torch.sum((1 - inputs) * targets)

        recall = (tp + eps) / (tp + fn + eps)

        return recall

    def __str__(self):
        return "recall"


class ReportMetrics:
    def __init__(self, model, metrics, device='cpu'):
        self.model = model
        self.metrics = metrics
        self.metrics_num = {k: [] for k in self.metrics.keys()}
        self.device = device

    def run_metrics(self, test_dataloader):
        metrics_batch_num = {k: [] for k in self.metrics.keys()}

        self.model = self.model.to(self.device)
        self.model.eval()

        for inputs, targets in tqdm(test_dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            _, predicted = self.model.val_on_batch(inputs, targets)

            for metric_name, _ in metrics_batch_num.items():
                metric_tensor = self.metrics[metric_name](predicted, targets)
                metrics_batch_num[metric_name].append(metric_tensor.item())

        for metric_name, metric_history in metrics_batch_num.items():
            self.metrics_num[metric_name].append(np.mean(metric_history))

        return self.metrics_num


"""
This module implements metrics realization for segmentation task
"""

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


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

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, smooth=1e-6) -> torch.Tensor:
        """
        Forward pass of IoU metric

        Args:
            outputs (torch.Tensor): Predicted inputs.
            targets (torch.Tensor): Target labels.
            smooth (float): Smoothing factor to avoid division by zero.

        Returns:
            torch.Tensor: score.

        """
        outputs = outputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (outputs * targets).sum()
        total = (outputs + targets).sum()
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

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, eps=1e-5) -> torch.Tensor:
        """
        Forward pass of dice metric

        Args:
            outputs (torch.Tensor): Predicted inputs.
            targets (torch.Tensor): Target labels.
            smooth (float): Smoothing factor to avoid division by zero.

        Returns:
            torch.Tensor: score.

        """
        outputs = outputs.reshape(-1)
        targets = targets.reshape(-1)

        sum = outputs.sum() + targets.sum()
        intersection = (outputs * targets).sum()
        dice = (2.0 * intersection + eps) / (sum + eps)
        return dice

    def __str__(self):
        return "dice"


class Accuracy(nn.Module):
    """
    Accuracy metric counts tp and tn pixels and divide this number by number of all pixels

    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, eps=1e-5) -> torch.Tensor:
        """
        Forward pass of accuracy metric.

        Args:
            outputs (torch.Tensor): Predicted inputs.
            targets (torch.Tensor): Target labels.
            eps (float): Smoothing factor to avoid division by zero.

        Returns:
            torch.Tensor: score.

        """
        tp = torch.sum(outputs * targets)
        fp = torch.sum(outputs * (1 - targets))
        fn = torch.sum((1 - outputs) * targets)
        tn = torch.sum((1 - outputs) * (1 - targets))

        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        return accuracy

    def __str__(self):
        return "accuracy"


class Precision(nn.Module):
    """
    Precision metric counts tp pixels and divide this number by all positive pixels
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, eps=1e-5) -> torch.Tensor:
        """
        Forward pass of precision metric.

        Args:
            outputs (torch.Tensor): Predicted inputs.
            targets (torch.Tensor): Target labels.
            eps (float): Smoothing factor to avoid division by zero.

        Returns:
            torch.Tensor: score.

        """
        tp = torch.sum(outputs * targets)
        fp = torch.sum(outputs * (1 - targets))

        precision = (tp + eps) / (tp + fp + eps)
        return precision

    def __str__(self):
        return "precision"


class Recall(nn.Module):
    """
    Recall metric counts tp pixels and divide this number by tp + fn pixels
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, eps=1e-5) -> torch.Tensor:
        """
        Forward pass of Recall metric.

        Args:
            outputs (torch.Tensor): Predicted inputs.
            targets (torch.Tensor): Target labels.
            eps (float): Smoothing factor to avoid division by zero.

        Returns:
            torch.Tensor: score.

        """
        tp = torch.sum(outputs * targets)
        fn = torch.sum((1 - outputs) * targets)

        recall = (tp + eps) / (tp + fn + eps)

        return recall

    def __str__(self):
        return "recall"


class TemporalConsistency(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.iou = IoUMetric()

    @staticmethod
    def warp_frame(frame, flow):
        _, _, height, width = frame.shape
        grid_x, grid_y = torch.meshgrid(torch.arange(height), torch.arange(width))
        grid_x = grid_x.float()
        grid_y = grid_y.float()

        # Add flow to grid
        flow_x = grid_x + flow[:, 0]
        flow_y = grid_y + flow[:, 1]

        # Normalize grid to [-1, 1]
        grid_normalized = torch.stack([(2 * flow_y / (flow_y.max() - 1)) - 1, (2 * flow_x / (flow_x.max() - 1)) - 1], dim=-1)

        # Perform sampling
        warped_frame = F.grid_sample(frame, grid_normalized, mode='bilinear', padding_mode='zeros')

        return warped_frame

    # TODO: remove image output
    def forward(
            self, frame_prev: torch.Tensor, frame_cur: torch.Tensor,
            mask_prev: torch.Tensor, mask_cur: torch.Tensor
    ): #-> torch.Tensor:
        # Calculating optical flow (check raft)
        raft_model = torchvision.models.optical_flow.raft_large(pretrained=True)
        with torch.no_grad():
            flow = raft_model(frame_prev.to(self.device), frame_cur.to(self.device))[5]

        # Calculating temporal consistency
        warped_frame = self.warp_frame(mask_prev, flow)
        temporal_consistency = self.iou(warped_frame, mask_cur)
        return temporal_consistency, warped_frame


class ReportMetrics:
    """
    Initialize the ReportMetrics class.

    Args:
        model (torch.nn.Module): The model to evaluate.
        metrics (dict): Dictionary containing metric names as keys and corresponding metric functions as values.
        device (str): Device to run the evaluation on. Default is 'cpu'.
    """
    def __init__(self, model: torch.nn.Module, metrics: dict, device='cpu'):
        self.model = model
        self.metrics = metrics
        self.metrics_num = {k: [] for k in self.metrics.keys()}
        self.device = device

    def run_metrics(self, test_dataloader: torch.utils.data.DataLoader) -> dict:
        """
        Run evaluation metrics on the test data.

        Args:
            test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.

        Returns:
            dict: Dictionary containing the computed metrics.
        """
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


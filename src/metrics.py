"""
This module implements metrics realization for segmentation task
"""
import torch
from torchmetrics import Metric, MetricCollection, Precision, Recall, F1Score


class IoUMetric(Metric):
    """
    Metric for binary segmentation using Intersection over Union (IoU)

    IoU is calculated as the ratio between the overlap of
    the positive instances between two sets, and their mutual combined values.

    J(A, B) = |A and B| / |A or B| = |A and B| / (|A| + |B| - |A and B|)
    """

    def __init__(self, eps=1e-6, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Add metric states
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.eps = eps

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Update the state with the current batch of predictions and targets.

        Args:
            outputs (torch.Tensor): Predicted outputs (binary values).
            targets (torch.Tensor): Ground truth labels (binary values).
        """
        outputs = outputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (outputs * targets).sum()
        total = (outputs + targets).sum()
        union = total - intersection

        # Update the states
        self.intersection += intersection
        self.union += union

    def compute(self) -> torch.Tensor:
        """
        Compute the final IoU score after accumulating over all batches.

        Returns:
            torch.Tensor: IoU score
        """
        return (self.intersection + self.eps) / (self.union + self.eps)

    def __str__(self):
        return "iou"


class DiceMetric(Metric):
    """
    Metric for binary segmentation using the Dice coefficient.

    The Dice coefficient is calculated as:
    Dice = 2 * (|A ∩ B|) / (|A| + |B|)
    """

    def __init__(self, eps=1e-5, dist_sync_on_step=False):
        """
        Args:
            eps (float): Smoothing factor to avoid division by zero.
            dist_sync_on_step (bool): Synchronize metric state across processes at each forward step (useful for DDP).
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Add metric states
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.eps = eps

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Update the state with the current batch of predictions and targets.

        Args:
            outputs (torch.Tensor): Predicted outputs (binary values).
            targets (torch.Tensor): Ground truth labels (binary values).
        """
        outputs = outputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (outputs * targets).sum()
        total = outputs.sum() + targets.sum()

        # Update the states
        self.intersection += intersection
        self.total += total

    def compute(self) -> torch.Tensor:
        """
        Compute the final Dice coefficient after accumulating over all batches.

        Returns:
            torch.Tensor: Dice score
        """
        return (2.0 * self.intersection + self.eps) / (self.total + self.eps)

    def __str__(self):
        return "dice"


def get_classification_metrics(**kwargs) -> MetricCollection:
    return MetricCollection({
        'f1': F1Score(**kwargs),
        'precision': Precision(**kwargs),
        'recall': Recall(**kwargs),
    })


def get_segmentation_metrics(**kwargs) -> MetricCollection:
    return MetricCollection({
        'iou': IoUMetric(**kwargs),
        'dice': DiceMetric(**kwargs),
    })

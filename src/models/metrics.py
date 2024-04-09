import torch
import torch.nn as nn

import torch
import numpy as np
from tqdm import tqdm
from src.models.utils.dirs import save_model
from src.features.adele import correct_mask, predict_average_on_scales
from src.features.adele_utils import create_labels_artifact, convert_data_to_dict, write_labels



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


class CosineSim(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask_1, mask_2):
        mask_1 = mask_1.reshape(-1)
        mask_2 = mask_2.reshape(-1)

        scalar_product = torch.sum(mask_1 * mask_2)
        mask_1_norm = torch.linalg.norm(mask_1, dim=0, ord=0)
        mask_2_norm = torch.linalg.norm(mask_2, dim=0, ord=0)

        cosine_sim = scalar_product / (mask_1_norm * mask_2_norm)
        return cosine_sim

    def __str__(self):
        return "cosine_sim"


class CosineUnionSim(nn.Module):
    def __init__(self):
        super().__init__()

        self.cosine_sim = CosineSim()

    def forward(self, mask_1, mask_2):
        mask_1 = mask_1.reshape(-1)
        mask_2 = mask_2.reshape(-1)
        mask_union = mask_1 + mask_2 - mask_1 * mask_2

        cosine_sim_1 = self.cosine_sim(mask_union, mask_1)
        cosine_sim_2 = self.cosine_sim(mask_union, mask_2)

        cosine_union_sim = 1.0/2.0 * (cosine_sim_1 + cosine_sim_2)
        return cosine_union_sim

    def __str__(self):
        return "cosine_union_sim"


class AverageMetric(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def forward(self, mask_1, target_1, mask_2, target_2):
        num_metric_1 = self.metric(mask_1, target_1)
        num_metric_2 = self.metric(mask_2, target_2)

        return 1.0/2.0 * (num_metric_1 + num_metric_2)

    def __str__(self):
        return "average_metric"


class CosineStability(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.average_metric = AverageMetric(self.metric)
        self.cosine_sim = CosineSim()

    def forward(self, mask_1, target_1, mask_2, target_2):
        cosine_sim_num = self.cosine_sim(mask_1, mask_2)
        average_metric_num = self.average_metric(mask_1, target_1, mask_2, target_2)
        cosine_stability = cosine_sim_num * average_metric_num
        return cosine_stability

    def __str__(self):
        return "cosine_stability"


class CosineUnionStability(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.average_metric = AverageMetric(self.metric)
        self.cosine_sim = CosineUnionSim()

    def forward(self, mask_1, target_1, mask_2, target_2):
        cosine_sim_num = self.cosine_sim(mask_1, mask_2)
        average_metric_num = self.average_metric(mask_1, target_1, mask_2, target_2)
        cosine_stability = cosine_sim_num * average_metric_num
        return cosine_stability

    def __str__(self):
        return "cosine_union_stability"


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


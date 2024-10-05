import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchmetrics
from src.metrics import IoUMetric


class TemporalConsistency(torchmetrics.Metric):
    """
    Metric for measuring temporal consistency between frames in video segmentation.

    This metric uses optical flow (RAFT model) to warp the previous frame's mask
    to the current frame and then calculates the IoU between the warped mask and the current mask.
    """

    def __init__(self, dist_sync_on_step=False):
        """
        Args:
            dist_sync_on_step (bool): Synchronize metric state across processes at each forward step (useful for DDP).
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Load RAFT optical flow model
        self.raft_model = torchvision.models.optical_flow.raft_large(pretrained=True)
        self.raft_model = self.raft_model.to(self.device)
        self.raft_model.eval()

        # IoU metric to calculate temporal consistency
        self.iou = IoUMetric()

        # Add states to accumulate over batches
        self.add_state("temporal_consistency_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def warp_frame(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp the previous frame's mask using the calculated optical flow.
        """
        _, _, height, width = frame.shape

        grid_x, grid_y = torch.meshgrid(torch.arange(height), torch.arange(width))
        grid_x = grid_x.float().to(self.device)
        grid_y = grid_y.float().to(self.device)
        flow_x = grid_x + flow[:, 0]
        flow_y = grid_y + flow[:, 1]

        # Normalize grid to [-1, 1]
        grid_normalized = torch.stack(
            [(2 * flow_y / (flow_y.max() - 1)) - 1, (2 * flow_x / (flow_x.max() - 1)) - 1], dim=-1
        )

        warped_frame = F.grid_sample(frame, grid_normalized, mode='bilinear', padding_mode='zeros')

        return warped_frame

    def update(
            self, frame_prev: torch.Tensor, frame_cur: torch.Tensor,
            mask_prev: torch.Tensor, mask_cur: torch.Tensor
    ):
        """
        Update the temporal consistency metric by calculating IoU between
        the warped previous mask and the current mask.

        Args:
            frame_prev (torch.Tensor): Previous frame (batch of images).
            frame_cur (torch.Tensor): Current frame (batch of images).
            mask_prev (torch.Tensor): Segmentation mask for the previous frame.
            mask_cur (torch.Tensor): Segmentation mask for the current frame.
        """
        # Calculate optical flow between previous and current frames
        with torch.no_grad():
            flow = self.raft_model(frame_prev.to(self.device), frame_cur.to(self.device))[5]

        # Warp the previous frame's mask
        warped_frame = self.warp_frame(mask_prev, flow)

        # Calculate IoU between warped mask and current mask
        temporal_consistency = self.iou(warped_frame, mask_cur)

        # Accumulate the temporal consistency value
        self.temporal_consistency_sum += temporal_consistency
        self.count += 1

    def compute(self) -> torch.Tensor:
        """
        Compute the average temporal consistency across all batches.

        Returns:
            torch.Tensor: The average temporal consistency.
        """
        return self.temporal_consistency_sum / self.count

    def __str__(self):
        return "temporal_consistency_raft"



class CosineSim(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def forward(self, tensor_1, tensor_2):
        """
        Takes tensors with shape (..., vec_dim), where vec_dim - dimension of the vectors,
        which cosine similarity we count
        """
        tensor_1 = tensor_1.to(self.device)
        tensor_2 = tensor_2.to(self.device)

        vec_dim = tensor_1.shape[-1]
        tensor_1 = torch.reshape(tensor_1, (-1, vec_dim))
        tensor_2 = torch.reshape(tensor_2, (-1, vec_dim))

        scalar_product = torch.sum(tensor_1 * tensor_2, dim=1)
        tensor_1_norm = torch.linalg.norm(tensor_1, dim=1)
        tensor_2_norm = torch.linalg.norm(tensor_2, dim=1)

        cosine_sim = torch.sum(scalar_product / (tensor_1_norm * tensor_2_norm)) / scalar_product.shape[0]
        return cosine_sim

    def __str__(self):
        return "cosine_sim"


class OpticalFlowSimilarity(torchvision.Metric):
    """
    Metric for calculating the similarity between optical flow of two sets of frames (e.g., frame pairs).

    This metric uses the RAFT model to compute the optical flow between two consecutive frames,
    and then calculates the cosine similarity between the resulting flow vectors.
    """

    def __init__(self, device='cpu', dist_sync_on_step=False):
        """
        Args:
            device (str): Device to perform the calculations on (e.g., 'cpu', 'cuda').
            dist_sync_on_step (bool): Synchronize metric state across processes at each forward step (useful for DDP).
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Load RAFT optical flow model
        self.device = device
        self.raft_model = torchvision.models.optical_flow.raft_small(
            weights=torchvision.models.optical_flow.Raft_Small_Weights.DEFAULT
        )
        self.raft_model = self.raft_model.to(self.device)
        self.raft_model.eval()

        # Initialize the cosine similarity metric
        self.cosine_sim = CosineSim(self.device)

        # Add states to accumulate over batches
        self.add_state("cosine_sim_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, frame_prev: torch.Tensor, frame_cur: torch.Tensor, mask_prev: torch.Tensor, mask_cur: torch.Tensor):
        """
        Update the metric state with new frames and masks to compute optical flow similarity.

        Args:
            frame_prev (torch.Tensor): Previous frame (batch of images).
            frame_cur (torch.Tensor): Current frame (batch of images).
            mask_prev (torch.Tensor): Previous frame's mask.
            mask_cur (torch.Tensor): Current frame's mask.
        """
        frame_prev, frame_cur = frame_prev.to(self.device), frame_cur.to(self.device)
        mask_prev, mask_cur = mask_prev.to(self.device), mask_cur.to(self.device)

        # Calculate optical flow between frames and masks using the RAFT model
        with torch.no_grad():
            flow = self.raft_model(frame_prev, frame_cur)[-1]
            mask_flow = self.raft_model(mask_prev, mask_cur)[-1]

        # Reshape the flow tensors
        b, v, h, w = flow.shape
        flow = torch.reshape(flow, (b, h, w, v))
        mask_flow = torch.reshape(mask_flow, (b, h, w, v))

        # Compute cosine similarity between the flow and mask flow
        cosine_sim_num = self.cosine_sim(flow, mask_flow)

        # Accumulate the cosine similarity and count
        self.cosine_sim_sum += cosine_sim_num
        self.count += 1

    def compute(self) -> torch.Tensor:
        """
        Compute the average cosine similarity across all batches.

        Returns:
            torch.Tensor: The average cosine similarity between optical flow and mask flow.
        """
        return self.cosine_sim_sum / self.count

    def __str__(self):
        return "optical_flow_similarity"


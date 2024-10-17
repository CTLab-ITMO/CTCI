"""
This module contains the implementation of a Segformer segmentation model
    and related functions for building the model.

"""
import torch
from torch import nn
from torch.nn import functional as func


class SegFormer(nn.Module):
    """
    SegFormer segmentation model.

    Args:
        net: SegFormer network architecture.
        mask_head: Layers for producing segmentation mask.
        loss_fn: Loss function for training the model.
        image_size (tuple): Input image size.
        device (str): Device for model computation.
    """
    def __init__(
            self, net, image_size=(256, 256)
    ):
        super().__init__()
        self.image_size = image_size
        self.net = net
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Segmentation mask tensor.
        """
        out = self.net(pixel_values=image, labels=None).logits
        out = self._interpolate_output(out)
        out = self.mask_head(out)
        return out

    def _interpolate_output(self, output) -> torch.Tensor:
        """
        Interpolates the output tensor to match the input image size.

        Args:
            output (torch.Tensor): Tensor representing the model output.

        Returns:
            torch.Tensor: Interpolated output tensor.
        """
        return func.interpolate(output, size=self.image_size, mode="bilinear", align_corners=False)

    def predict(self, image: torch.Tensor, conf=0.6) -> torch.Tensor:
        """
        Performs predict on an input image.

        Args:
            image (torch.Tensor): Input image tensor.
            conf (float): Confidence threshold for binarizing the output mask.

        Returns:
            torch.Tensor: Predicted segmentation mask tensor.
        """
        out = self.forward(image=image)
        out = torch.where(out > conf, 1, 0)
        return out

    def __str__(self):
        return "segformer"

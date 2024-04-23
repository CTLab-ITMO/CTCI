"""
This module contains the implementation of a Segformer segmentation model
    and related functions for building the model.

"""
import torch
import torch.nn as nn

import transformers

from src.models.base_model import BaseModel
from src.models.utils.config import ConfigHandler


class SegFormer(BaseModel):
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
            self, net, mask_head=None, loss_fn=None,
            image_size=(256, 256), device="cpu"
    ):
        super().__init__()

        self.device = device
        self.image_size = image_size
        self.net = net.to(self.device)

        if mask_head:
            self.mask_head = mask_head.to(self.device)
        else:
            self.mask_head = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            ).to(self.device)

        if loss_fn:
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = nn.BCELoss().to(self.device)

    def forward(self, image: torch.Tensor, labels=None) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Segmentation mask tensor.
        """
        out = self.net(pixel_values=image, labels=labels).logits
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
        return nn.functional.interpolate(output, size=self.image_size, mode="bilinear", align_corners=False)

    def _calc_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss function.

        Args:
            output (torch.Tensor): Predicted segmentation mask tensor.
            target (torch.Tensor): Ground truth segmentation mask tensor.

        Returns:
            torch.Tensor: Loss tensor.
        """
        return self.loss_fn(output, target)

    def train_on_batch(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Performs a training iteration on a batch of data.

        Args:
            image (torch.Tensor): Input image tensor.
            target (torch.Tensor): Ground truth segmentation mask tensor.

        Returns:
            torch.Tensor: Loss tensor.
        """
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, labels)
        return loss

    def val_on_batch(self, image: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Performs evaluation on a batch of data.

        Args:
            image (torch.Tensor): Input image tensor.
            target (torch.Tensor): Ground truth segmentation mask tensor.

        Returns:
            torch.Tensor: Loss tensor.
            torch.Tensor: Predicted segmentation mask tensor.
        """
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, labels)
        return loss, outputs

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


def build_segformer(config_handler: ConfigHandler):
    """
    Builds a SegFormer segmentation model based on configuration file.

    Args:
        config_handler (ConfigHandler): Configuration file handler.

    Returns:
        SegFormer: SegFormer segmentation model.
    """
    device = config_handler.read('model', 'device')

    model_name = config_handler.read('model', 'model_name')
    model_type = config_handler.read('model', 'model_type')

    image_size_width = config_handler.read('dataset', 'image_size', 'width')
    image_size_height = config_handler.read('dataset', 'image_size', 'height')
    image_size = (image_size_width, image_size_height)

    net = transformers.SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/{model_name}-{model_type}-finetuned-ade-512-512",
        num_labels=1,
        image_size=image_size_height,
        ignore_mismatched_sizes=True
    )
    final_layer = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()

    segformer = SegFormer(
        net=net, mask_head=final_layer, loss_fn=loss_fn,
        image_size=image_size, device=device
    )

    return segformer

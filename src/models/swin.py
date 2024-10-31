"""
This module contains the implementation of a Swin (+UNETR) segmentation model
    and related functions for building the model.

"""
import torch
from torch import nn
from src.models.unetr import UNETRDecoder


class Swin(nn.Module):
    """
    Swin segmentation model.

    Args:
        net: Swin network architecture.
        mask_head: UNETRDecoder.
        loss_fn: Loss function for training the model.
        image_size (tuple): Input image size.
        device (str): Device for model computation.

    """
    def __init__(
            self, net
    ):
        super().__init__()

        self.net = net.encoder
        self.mask_head = UNETRDecoder()
        self.embeddings = net.get_input_embeddings()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Segmentation mask tensor.
        """
        emb, input_dim = self.embeddings(image)
        out = self.net(
            hidden_states=emb,
            input_dimensions=input_dim,
            output_hidden_states=True
        )

        out = self.mask_head(
            reshaped_hidden_states=out.reshaped_hidden_states,
            image=image
        )
        return out

    def predict(self, image: torch.Tensor, conf=0.6) -> torch.Tensor:
        """
        Performs predict on an input image.

        Args:
            image (torch.Tensor): Input image tensor.
            conf (float): Confidence threshold for binarizing the output mask.

        Returns:
            torch.Tensor: Predicted segmentation mask tensor.
        """
        out = self.forward(image)
        out = torch.where(out > conf, 1, 0)
        return out

    def __str__(self):
        return "swin"


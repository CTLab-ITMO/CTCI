"""
This module contains the implementation of UNETR Decoder architecture.

"""
from typing import List

import torch

from src.models.blocks import *


class UNETRDecoder(nn.Module):
    """
    Decoder module for the UNETR architecture.

    Args:
        in_channels (int): Number of input channels. Default is 3.
        out_channels (int): Number of output channels. Default is 1.
        feature_size (int): Size of the feature maps. Default is 96.
        hidden_size (List[int]): List containing sizes of hidden layers. Default is [192, 384, 768, 768].
    """
    def __init__(self, in_channels=3, out_channels=1, feature_size=96, hidden_size=[192, 384, 768, 768]):
        super().__init__()

        self.features = ResBlock(in_channels, feature_size)
        self.patch_features = RepresentationBlock(
            feature_size,
            feature_size
        )
        self.encoder1 = RepresentationBlock(
            hidden_size[0],
            feature_size * 2
        )
        self.encoder2 = RepresentationBlock(
            hidden_size[1],
            feature_size * 4
        )
        self.encoder3 = RepresentationBlock(
            hidden_size[2],
            feature_size * 8
        )
        self.bottleneck = ConvBlock(
            hidden_size[3],
            feature_size * 8
        )
        self.decoder3 = DecoderBlock(
            feature_size * 8,
            feature_size * 8
        )
        self.decoder2 = DecoderBlock(
            feature_size * 8,
            feature_size * 4
        )
        self.decoder1 = DecoderBlock(
            feature_size * 4,
            feature_size * 2
        )

        self.decoder0 = DecoderBlock(
            feature_size * 2,
            feature_size
        )

        self.patch_decoder = DecoderBlock(
            feature_size,
            feature_size
        )

        self.final_layer = nn.Sequential(
            ConvBlock(
                feature_size,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            ConvBlock(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm=False,
                act=False
            )
        )
        self.act = nn.Sigmoid()

    def forward(self, reshaped_hidden_states: List[torch.Tensor], image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            reshaped_hidden_states (List[torch.Tensor]): List of hidden states.
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        im = self.features(image)

        p = self.patch_features(reshaped_hidden_states[0])
        
        state1 = self.encoder1(reshaped_hidden_states[1])
        state2 = self.encoder2(reshaped_hidden_states[2])
        state3 = self.encoder3(reshaped_hidden_states[3])
        bottleneck = self.bottleneck(reshaped_hidden_states[4])

        dec3 = self.decoder3(bottleneck, state3)
        dec2 = self.decoder2(dec3, state2)
        dec1 = self.decoder1(dec2, state1)

        pd = self.decoder0(dec1, p)
        out = self.patch_decoder(pd, im)
        out = self.final_layer(out)

        out = self.act(out)

        return out

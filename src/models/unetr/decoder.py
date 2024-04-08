import torch
import numpy as np
import torch.nn as nn
from src.models.unetr.blocks import *


class UNETRDecoder(nn.Module):
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

    def forward(self, reshaped_hidden_states, image):

        im = self.features(image)

        p = self.patch_features(reshaped_hidden_states[0])
        
        state1 = self.encoder1(reshaped_hidden_states[1])
        state2 = self.encoder2(reshaped_hidden_states[2])
        state3 = self.encoder3(reshaped_hidden_states[3])
        btlnck = self.bottleneck(reshaped_hidden_states[4])

        dec3 = self.decoder3(btlnck, state3)
        dec2 = self.decoder2(dec3, state2)
        dec1 = self.decoder1(dec2, state1)

        pd = self.decoder0(dec1, p)
        out = self.patch_decoder(pd, im)
        out = self.final_layer(out)

        out = self.act(out)

        return out

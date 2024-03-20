import torch
import numpy as np
import torch.nn as nn
from src.models.unetr.blocks import *


class UNETRDecoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_size=96, hidden_size=[96, 192, 384, 768]):
        super().__init__()

        self.feature = ResBlock(in_channels, feature_size)
        self.encoder1 = RepresentationBlock(
            hidden_size[0],
            feature_size
        )
        self.encoder2 = RepresentationBlock(
            hidden_size[1],
            feature_size * 2
        )
        self.encoder3 = RepresentationBlock(
            hidden_size[2],
            feature_size * 4,
            num_layers=0
        )
        self.bottleneck = DeconvBlock(
            hidden_size[3],
            feature_size * 8
        )
        self.decoder3 = DecoderBlock(
            feature_size * 8,
            feature_size * 4
        )
        self.decoder2 = DecoderBlock(
            feature_size * 4,
            feature_size * 2
        )
        self.decoder1 = DecoderBlock(
            feature_size * 2,
            feature_size
        )

        self.decoder0 = DecoderBlock(
            feature_size,
            feature_size
        )

        self.final_layer = ConvBlock(
            feature_size,
            out_channels,
            norm=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, reshaped_hidden_states, x):

        im = self.feature(x)
        
        state1 = self.encoder1(reshaped_hidden_states[0])
        state2 = self.encoder2(reshaped_hidden_states[1])
        state3 = self.encoder3(reshaped_hidden_states[2])
        btlnck = self.bottleneck(reshaped_hidden_states[3])

        dec3 = self.decoder3(btlnck, state3)
        dec2 = self.decoder2(dec3, state2)
        dec1 = self.decoder1(dec2, state1)

        out = self.decoder0(dec1, im)
        out = self.final_layer(out)

        return self.sigmoid(out)

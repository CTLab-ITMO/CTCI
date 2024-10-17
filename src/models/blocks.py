"""
This module contains various building blocks used in UNETRDecoder.

"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional block consisting of a convolutional layer, optional instance normalization, and ReLU activation.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel. Default is 3.
        padding (int): Amount of padding. Default is 1.
        stride (int): Stride of the convolution. Default is 1.
        norm (bool): Whether to apply instance normalization. Default is True.
        act (bool): Whether to apply ReLU activation. Default is True.
    """
    def __init__(self, in_c: int, out_c: int, kernel_size=3, padding=1, stride=1, norm=True, act=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_c, out_c,
                      kernel_size=kernel_size, padding=padding, stride=stride)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_c))
        if act:
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return self.conv(x)


class DeconvBlock(nn.Module):
    """
    Deconvolutional block consisting of a transpose convolutional layer followed by ReLU activation.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return self.act(self.deconv(x))


class ResBlock(nn.Module):
    """
    Residual block consisting of two convolutional blocks and a skip connection.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()

        self.conv1 = ConvBlock(in_c, out_c)
        self.conv2 = None
        self.act = nn.ReLU()

        self.downsample = in_c != out_c
        if self.downsample:
            self.conv2 = ConvBlock(in_c, out_c, kernel_size=1, padding=0, norm=False, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        residual = x
        out = self.conv1(x)
        if self.conv2:
            residual = self.conv2(residual)
        res = self.act(residual + out)
        return res


class RepresentationBlock(nn.Module):
    """
    Representation block consisting of an upsampling operation followed by multiple convolutional or residual blocks.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        res (bool): Whether to use residual blocks. Default is True.
        num_layers (int): Number of convolutional or residual blocks. Default is 1.
    """
    def __init__(self, in_c: int, out_c: int, res=True, num_layers=1):
        super().__init__()

        self.upsample = DeconvBlock(in_c, out_c)
        if res:
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        ResBlock(out_c, out_c)
                    ) for _ in range(num_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvBlock(out_c, out_c)
                    ) for _ in range(num_layers)
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        out = self.upsample(x)
        for block in self.blocks:
            out = block(out)
        return out


class DecoderBlock(nn.Module):
    """
    Decoder block consisting of an upsampling operation followed by a convolutional or residual block.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        res (bool): Whether to use residual blocks. Default is True.
    """
    def __init__(self, in_c: int, out_c: int, res=True):
        super().__init__()

        self.upsample = DeconvBlock(in_c, out_c)

        if res:
            self.conv = ResBlock(out_c + out_c, out_c)
        else:
            self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            skip (torch.Tensor): Skip connection tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        out = self.upsample(x)
        out = torch.cat((out, skip), dim=1)
        out = self.conv(out)
        return out


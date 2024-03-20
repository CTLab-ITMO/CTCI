import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, stride=1, norm=True, act=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_c, out_c,
                      kernel_size=kernel_size, padding=padding, stride=stride)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        if act:
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.deconv(x))


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = ConvBlock(in_c, out_c)
        self.conv2 = None
        self.act = nn.ReLU()

        self.downsample = in_c != out_c
        if self.downsample:
            self.conv2 = ConvBlock(in_c, out_c, kernel_size=1, padding=0, norm=False, act=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.conv2:
            residual = self.conv2(residual)
        res = self.act(residual + out)
        return res


class RepresentationBlock(nn.Module):
    def __init__(self, in_c, out_c, res=True, num_layers=1):
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

    def forward(self, x):
        out = self.upsample(x)
        for block in self.blocks:
            out = block(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, res=True):
        super().__init__()

        self.upsample = DeconvBlock(in_c, out_c)

        if res:
            self.conv = ResBlock(out_c + out_c, out_c)
        else:
            self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, x, skip):
        out = self.upsample(x)
        out = torch.cat((out, skip), dim=1)
        out = self.conv(out)
        return out


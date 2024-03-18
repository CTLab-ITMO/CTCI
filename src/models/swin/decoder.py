import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, norm=True, act=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_c, out_c,
                      kernel_size=kernel_size, padding=padding)
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
            in_c=in_c,
            out_channels=out_c,
            kernel_size=2,
            stride=2,
            padding=0
        )

    def forward(self, x):
        return self.deconv(x)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = ConvBlock(in_c, out_c)
        self.conv2 = ConvBlock(out_c, out_c)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        res = self.act(x + out)
        return res


class RepresentationBlock(nn.Module):
    def __init__(self, in_c, out_c, res=True, num_layers=2):
        super().__init__()

        self.upsample = DeconvBlock(in_c, out_c)
        if res:
            self.blocks = nn.Sequential(
                [
                    DeconvBlock(out_c, out_c),
                    ResBlock(out_c, out_c)
                ] * num_layers
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        DeconvBlock(out_c, out_c),
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
            self.conv = ResBlock(out_c+out_c, out_c)
        else:
            self.conv = ConvBlock(out_c+out_c, out_c)

    def forward(self, x, skip):
        out = self.upsample(x)
        out = torch.cat((out, skip), dim=1)
        out = self.conv(out)
        return out


class UNETRDecoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_size=48, hidden_size=[48, 96, 192, 384]):
        super().__init__()

        self.feature = ResBlock(in_channels, feature_size)
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




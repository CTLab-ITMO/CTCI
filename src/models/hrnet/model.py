import torch
import torch.nn as nn
import torch.nn.functional as F


class HRNetModel(nn.Module):
    def __init__(self, net=None, image_size=(256, 256)):
        super().__init__()

        self.net = net
        self.image_size = image_size
        total_num_features = sum(self.net.feature_info.channels())

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=total_num_features, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        out = self.net(x)
        interpolated = []
        for o in out:
            interpolated.append(self._interpolate_output(o))
        out = torch.cat(interpolated, axis=1)
        out = self.last_layer(out)
        return out

    def _interpolate_output(self, out):
        h, w = self.image_size
        return F.interpolate(input=out, size=(h, w), mode='bilinear', align_corners=True)


    def calc_loss_fn(self, output, target):
        # badly written code just to start training
        return self.loss_fn(output, target)

    def train_on_batch(self, input, target):
        out = self.forward(input)
        loss = self.calc_loss_fn(out, target)
        return loss
    
    def val_on_batch(self, image, target):
        out = self.forward(image)
        loss = self.calc_loss_fn(out, target)

        return loss, out

    def predict(self, x):
        out = self.forward(x)
        return out
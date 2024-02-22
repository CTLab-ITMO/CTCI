import timm
import torch.nn as nn
from typing import Union, Dict


class HRNet(nn.Module):

    def __init__(self, hrnet_type="hrnet_w30", only_classifier=True):
        super().__init__()

        self.net = timm.create_model(
            hrnet_type,
            pretrained=True,
            features_only=True,
            num_classes=1,
            out_indices=(0,1,2)
            )
        self.cls = self._build_classifier()

        if only_classifier:
            self.freeze_backbone()

    def _build_classifier(self, in_channels=5):
        cls = [
            nn.Conv2d(
                in_channels=in_channels,  # for pretrained hrnet
                out_channels=1,
                kernel_size=1),
            nn.Sigmoid()
        ]
        return nn.Sequential(*cls)

    def forward(self, x):
        features = self.net(x)
        out = self.classification(features)
        return out

    def set_loss_fn(self, loss_fn: Dict[nn.Module, float]):
        self.loss_fn = loss_fn

    def calc_loss_fn(self, input, target):
        l = 0
        for loss, weight in self.loss_fn.items():
            l += weight * loss(input, target)
        return l

    def freeze_backbone(self):
        self.net.eval()

    def train_on_batch(self, images, target):
        out = self.forward(images)
        loss = self.calc_loss_fn(out, target)
        return loss

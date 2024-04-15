import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from src.models.base_model import BaseModel
from src.models.utils.config import ConfigHandler


class HRNetModel(BaseModel):
    def __init__(
            self, net, mask_head=None, loss_fn=None,
            image_size=(256, 256), device="cpu"
    ):
        super().__init__()

        self.device = device
        self.image_size = image_size

        self.net = net.to(self.device)
        total_num_features = sum(self.net.feature_info.channels())

        if mask_head:
            self.mask_head = mask_head.to(self.device)
        else:
            self.mask_head = nn.Sequential(
                nn.Conv2d(in_channels=total_num_features, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            ).to(self.device)

        if loss_fn:
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = nn.BCELoss().to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        interpolated = []
        for o in out:
            interpolated.append(self._interpolate_output(o))
        out = torch.cat(interpolated, axis=1)
        out = self.mask_head(out)
        return out

    def _interpolate_output(self, out: torch.Tensor):
        h, w = self.image_size
        return F.interpolate(input=out, size=(h, w), mode='bilinear', align_corners=True)

    def calc_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: badly written code just to start training, need to refactor
        return self.loss_fn(output, target)

    def train_on_batch(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = self.forward(image)
        loss = self.calc_loss_fn(out, target)
        return loss
    
    def val_on_batch(self, image: torch.Tensor, target: torch.Tensor):
        out = self.forward(image)
        loss = self.calc_loss_fn(out, target)
        return loss, out

    def predict(self, image: torch.Tensor, conf=0.6) -> torch.Tensor:
        out = self.forward(image)
        out = torch.where(out > conf, 1, 0)
        return out


def build_hrnet(config_handler: ConfigHandler):
    device = config_handler.read('model', 'device')

    model_name = config_handler.read('model', 'model_name')

    image_size_width = config_handler.read('dataset', 'image_size', 'width')
    image_size_height = config_handler.read('dataset', 'image_size', 'height')
    image_size = (image_size_width, image_size_height)

    net = timm.create_model(model_name, features_only=True, pretrained=True)
    loss_fn = nn.BCELoss()
    total_num_features = sum(net.feature_info.channels())
    mask_head = nn.Sequential(
        nn.Conv2d(in_channels=total_num_features, out_channels=1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid(),
        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
    )

    hrnet = HRNetModel(
        net=net, mask_head=mask_head, loss_fn=loss_fn,
        image_size=image_size, device=device
    )

    return hrnet

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

from src.models.base_model import BaseModel
from src.models.loss.soft_dice_loss import SoftDiceLossV2
from src.models.utils.config import ConfigHandler


class DeepLab(BaseModel):
    def __init__(
            self, net, mask_head=None, loss_fn=None,
            image_size=(256, 256), device="cpu"
    ):
        super().__init__()

        self.device = device
        self.image_size = image_size

        self.net = net.to(self.device)

        if mask_head:
            self.final_layer = mask_head.to(self.device)
        else:
            self.final_layer = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            ).to(self.device)

        if loss_fn:
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = SoftDiceLossV2().to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.net(image)
        out = self.final_layer(out)
        return out

    def _calc_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(output, target)

    def train_on_batch(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, target)
        return loss

    def val_on_batch(self, image: torch.Tensor, target: torch.Tensor):
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, target)
        return loss, outputs

    def predict(self, image: torch.Tensor, conf=0.6) -> torch.Tensor:
        out = self.forward(image)
        out = torch.where(out > conf, 1, 0)
        return out

    def __str__(self):
        return "deeplabv3"


def build_deeplab(config_handler: ConfigHandler):
    device = config_handler.read('model', 'device')

    model_name = config_handler.read('model', 'model_name')

    image_size_width = config_handler.read('dataset', 'image_size', 'width')
    image_size_height = config_handler.read('dataset', 'image_size', 'height')
    image_size = (image_size_width, image_size_height)

    net = smp.DeepLabV3Plus(encoder_name=model_name)
    final_layer = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
    )
    loss_fn = SoftDiceLossV2()

    deeplab = DeepLab(
        net=net, mask_head=final_layer, loss_fn=loss_fn,
        image_size=image_size, device=device
    )

    return deeplab


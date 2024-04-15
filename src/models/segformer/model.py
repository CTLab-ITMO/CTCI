import torch
import torch.nn as nn

import transformers

from src.visualization.visualization import restore_image_from_logits
from src.models.utils.config import ConfigHandler


class SegFormer(nn.Module):
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
            self.loss_fn = nn.BCELoss().to(self.device)

    def forward(self, pixel_values, labels=None):
        out = self.net(pixel_values=pixel_values, labels=labels).logits
        out = self._interpolate_output(out)
        out = self.final_layer(out)
        return out

    def _interpolate_output(self, out):
        return nn.functional.interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)

    def _calc_loss_fn(self, output, target):
        return self.loss_fn(output, target)

    def train_on_batch(self, pixel_values, labels):
        outputs = self.forward(pixel_values)
        loss = self._calc_loss_fn(outputs, labels)
        return loss

    def val_on_batch(self, pixel_values, labels):
        outputs = self.forward(pixel_values)
        loss = self._calc_loss_fn(outputs, labels)
        return loss, outputs

    def predict(self, image, conf=0.6):
        out = self.forward(pixel_values=image)
        out = torch.where(out > conf, 1, 0)
        return out

    def __str__(self):
        return "segformer"


def build_segformer(config_handler: ConfigHandler):
    device = config_handler.read('model', 'device')

    model_name = config_handler.read('model', 'model_name')
    model_type = config_handler.read('model', 'model_type')

    image_size_width = config_handler.read('dataset', 'image_size', 'width')
    image_size_height = config_handler.read('dataset', 'image_size', 'height')
    image_size = (image_size_width, image_size_height)

    net = transformers.SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/{model_name}-{model_type}-finetuned-ade-512-512",
        num_labels=1,
        image_size=image_size_height,
        ignore_mismatched_sizes=True
    )
    final_layer = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()

    segformer = SegFormer(
        net=net, mask_head=final_layer, loss_fn=loss_fn,
        image_size=image_size, device=device
    )

    return segformer

import torch
import torch.nn as nn

from src.visualization.visualization import restore_image_from_logits


class SegFormer(nn.Module):
    def __init__(self, net, image_size=(256, 256), device="cpu"):
        super().__init__()
        self.device = device
        self.image_size = image_size

        self.net = net.to(device)
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()

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

    def predict(self, image):
        # pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        # pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.forward(pixel_values=image)
        out = torch.where(outputs > 0.6, 1, 0)
        return out
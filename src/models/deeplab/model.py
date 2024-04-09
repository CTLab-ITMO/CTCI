import torch
import torch.nn as nn

from src.models.loss.soft_dice_loss import SoftDiceLossV2


class DeepLab(nn.Module):
    def __init__(self, net, device="cpu"):
        super().__init__()
        self.device = device
        self.net = net.to(device)
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.loss_fn = SoftDiceLossV2()

    def forward(self, image):
        out = self.net(image)
        out = self.final_layer(out)
        return out

    def _calc_loss_fn(self, output, target):
        return self.loss_fn(output, target)

    def train_on_batch(self, image, target):
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, target)
        return loss

    def val_on_batch(self, image, target):
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, target)
        return loss, outputs

    def predict(self, image):
        # pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        # pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            out = self.forward(image)
        out = torch.where(out > 0.6, 1, 0)
        return out
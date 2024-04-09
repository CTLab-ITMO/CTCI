import torch
import numpy as np
import torch.nn as nn
from src.models.unetr.decoder import UNETRDecoder
from src.models.loss.soft_dice_loss import SoftDiceLossV2

class Swin(nn.Module):
    def __init__(self, net, device="cpu", freeze_backbone=False):
        super().__init__()
        self.encoder = net.encoder.to(device)
        self.decoder = UNETRDecoder()
        self.embeddings = net.get_input_embeddings()
        self.loss_fn = SoftDiceLossV2()

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, image):
        emb, input_dim = self.embeddings(image)
        out = self.encoder(
            hidden_states=emb,
            input_dimensions=input_dim,
            output_hidden_states=True
        )

        out = self.decoder(
            reshaped_hidden_states=out.reshaped_hidden_states,
            image=image
        )
        return out

    def freeze_backbone(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

    def _calc_loss_fn(self, image, target):
        return self.loss_fn(image, target)

    def train_on_batch(self, image, target):
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, target)
        return loss

    def val_on_batch(self, image, target):
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, target)
        return loss, outputs

    def predict(self, image):
        out = self.forward(image)
        out = torch.where(out > 0.6, 1, 0)
        return out

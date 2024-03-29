import torch
import torch.nn as nn
from typing import Dict
from src.models.hrnet.hrnet_source.arch import get_seg_model
from src.models.hrnet.hrnet_source.criterion import CrossEntropy
from bestconfig import Config

config = Config("../src/models/hrnet/hrnet_source/cfg.yaml")


class HRNet(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()

        self.net = get_seg_model(config)
        self.last_layer = nn.Sigmoid()
        self.loss_fn = CrossEntropy()

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        out_aux, out = self.net(x)

        # out_aux = self.last_layer(out_aux)
        # out = self.last_layer(out)
        return [out_aux, out]
    
    def predict(self, x):
        _, out = self.net(x)
        return out

    def calc_loss_fn(self, output, target):
        # badly written code just to start training
        return self.loss_fn(output, target)

    def freeze_backbone(self):
        keys = ('cls_head', 'aux_head')
        for name, param in self.net.named_parameters():
            if not name.startswith(keys):
                param.requires_grad = False

    def train_on_batch(self, input, target):
        out = self.forward(input)
        loss = self.calc_loss_fn(out, target)
        return loss
    
    def val_on_batch(self, image, target):
        out = self.forward(image)
        loss = self.calc_loss_fn(out, target)
        predicted_mask = out[1]

        return loss, predicted_mask

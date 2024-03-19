import torch
import numpy as np
import torch.nn as nn
from src.models.swin.decoder import UNETRDecoder


class Swin(nn.Module):
    def __init__(self, net, image_processor=None, device="cpu"):
        super().__init__()
        self.device = device
        self.encoder = net.to(device)
        self.decoder = UNETRDecoder()
        self.image_processor = image_processor
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image):
        out = self.encoder(pixel_values=image, output_hidden_states=True)
        out = self.decoder(out.reshaped_hidden_states, image)
        return out

    def _calc_loss_fn(self, image, target):
        return self.loss_fn(image, target)

    def train_on_batch(self, image, target):
        outputs = self.forward(image)
        loss = self._calc_loss_fn(outputs, target)
        return outputs

    def val_on_batch(self, pixel_values, labels):
        outputs = self.forward(pixel_values, labels)
        logits, loss = outputs.logits, outputs.loss

        predicted_mask = self._restore_image_from_logits(
            logits, labels.shape[-2:])
        return loss, predicted_mask

    def predict(self, image):
        pixel_values = self.image_processor(
            image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.net(pixel_values=pixel_values)

        assert self.image_processor is not None, "image processor was missed"
        predicted_segmentation_map = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        return predicted_segmentation_map

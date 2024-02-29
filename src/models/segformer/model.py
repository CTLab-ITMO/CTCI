import torch
import torch.nn as nn


class SegFormer(nn.Module):
    def __init__(self, net, image_processor=None, device="cpu"):
        super().__init__()
        self.device = device
        self.net = net.to(device)
        self.image_processor = image_processor

    def _restore_image_from_logits(self, logits, size):
        upsampled_logits = nn.functional.interpolate(logits, size=size, mode="bilinear",
                                                     align_corners=False)
        predicted_mask = upsampled_logits.argmax(dim=1)
        return predicted_mask

    def forward(self, pixel_values, labels):
        out = self.net(pixel_values=pixel_values, labels=labels)
        return out

    def train_on_batch(self, pixel_values, labels):
        outputs = self.forward(pixel_values, labels)
        loss = outputs.loss
        return loss

    def val_on_batch(self, pixel_values, labels):
        outputs = self.forward(pixel_values, labels)
        logits, loss = outputs.logits, outputs.loss

        predicted_mask = self._restore_image_from_logits(logits, labels.shape[-2:])
        return loss, predicted_mask

    def predict(self, image):
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.net(pixel_values=pixel_values)

        assert self.image_processor is not None, "image processor was missed"
        predicted_segmentation_map = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        return predicted_segmentation_map

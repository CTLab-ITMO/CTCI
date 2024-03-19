import torch.nn as nn


def restore_image_from_logits(logits, size):
    upsampled_logits = nn.functional.interpolate(logits, size=size, mode="bilinear",
                                                 align_corners=False)
    predicted_mask = upsampled_logits.argmax(dim=1)
    return predicted_mask


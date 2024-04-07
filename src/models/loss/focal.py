import torch
import torch.nn as nn

class FocalLossBin(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, smooth=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        criterion = nn.BCELoss(reduction='mean')

        bce = criterion(outputs, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce

        return focal_loss

    def __repr__(self):
        description = f"Focal loss for binary segmentation"
        return description
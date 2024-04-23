"""
This module implements neural networks losses

"""
# TODO: Дошик, напиши сверху, откуда ты код софт дайса взяла
# TODO: Чаще всего по лицензиям надо указывать это
import torch
import torch.nn as nn
import torch.cuda.amp as amp


class FocalLossBin(nn.Module):
    """
    Focal loss for binary segmentation.

    Args:
        alpha (float): Weighting factor for positive samples. Default is 0.8.
        gamma (float): Focusing parameter. Default is 2.
        smooth (float): Smoothing factor to avoid division by zero. Default is 1.

    Returns:
        torch.Tensor: Focal loss.
    """
    def __init__(self, alpha=0.8, gamma=2, smooth=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss

        inputs:
            outputs (torch.Tensor): tensor of shape (N, H, W, ...)
            targets (torch.Tensor): tensor of shape(N, H, W, ...)
        output:
            loss (torch.Tensor): tensor of shape(1, )
        """
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


class SoftDiceLossV1(nn.Module):
    """
    Soft-Dice loss for binary segmentation.

    Args:
        p (float): Power parameter. Default is 1.
        smooth (float): Smoothing factor to avoid division by zero. Default is 1.

    Returns:
        torch.Tensor: Soft-Dice loss.
    """
    def __init__(
            self,
            p=1,
            smooth=1
    ):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss

        inputs:
            outputs (torch.Tensor): tensor of shape (N, H, W, ...)
            targets (torch.Tensor): tensor of shape(N, H, W, ...)
        output:
            loss (torch.Tensor): tensor of shape(1, )
        """
        probs = torch.sigmoid(outputs)
        numer = (probs * targets).sum()
        denor = (probs.pow(self.p) + targets.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss

    def __repr__(self):
        description = f"Soft-Dice loss for binary segmentation."
        return description


class SoftDiceLossV2(nn.Module):
    """
    Soft-Dice loss for binary segmentation with custom gradient formula.

    Args:
        p (float): Power parameter. Default is 1.
        smooth (float): Smoothing factor to avoid division by zero. Default is 1.

    Returns:
        torch.Tensor: Soft-Dice loss.
    """
    def __init__(
            self,
            p=1,
            smooth=1
    ):
        super(SoftDiceLossV2, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss

        inputs:
            outputs (torch.Tensor): tensor of shape (N, H, W, ...)
            targets (torch.Tensor): tensor of shape(N, H, W, ...)
        output:
            loss (torch.Tensor): tensor of shape(1, )
        """
        outputs = outputs.view(1, -1)
        targets = targets.view(1, -1)
        loss = SoftDiceLossV2Func.apply(outputs, targets, self.p, self.smooth)
        return loss

    def __repr__(self):
        description = f"Soft-Dice loss for binary segmentation with custom gradient formula."
        return description


class SoftDiceLossV2Func(torch.autograd.Function):
    """
    Custom autograd function for computing Soft-Dice loss backward pass directly.

    This function computes both the forward and backward passes for the Soft-Dice loss
    with a custom gradient formula for better numeric stability.

    """
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, probs: torch.Tensor, labels: torch.Tensor, p: float, smooth: float) -> torch.Tensor:
        """
        Forward pass of the Soft-Dice loss.

        Args:
            ctx: Context.
            probs (torch.Tensor): Predicted probabilities.
            labels (torch.Tensor): Target labels.
            p (float): Power parameter.
            smooth (float): Smoothing factor.

        Returns:
            torch.Tensor: Soft-Dice loss.

        """
        #  logits = logits.float()

        # probs = torch.sigmoid(logits)
        numer = 2 * (probs * labels).sum(dim=1) + smooth
        denor = (probs.pow(p) + labels.pow(p)).sum(dim=1) + smooth
        loss = 1. - numer / denor

        ctx.vars = probs, labels, numer, denor, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass of the Soft-Dice loss.

        Args:
            ctx: Context.
            grad_output (torch.Tensor): Gradient of the loss.

        Returns:
            tuple: Gradients with respect to inputs (probs, labels).

        """
        probs, labels, numer, denor, p, smooth = ctx.vars
        numer, denor = numer.view(-1, 1), denor.view(-1, 1)

        term1 = (1. - probs).mul_(2).mul_(labels).mul_(probs).div_(denor)
        term2 = probs.pow(p).mul_(1. - probs).mul_(numer).mul_(p).div_(denor.pow_(2))
        grads = term2.sub_(term1).mul_(grad_output)

        return grads, None, None, None

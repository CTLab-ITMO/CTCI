"""
This module implements the BaseModel interface
"""
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for neural network models.

    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    def _calc_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss function.

        Args:
            output (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Loss tensor.
        """
        pass

    def train_on_batch(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        This method implements a part of training loop
        which is where the model gives output and then the
        loss is calculated

        training loop usually is structured like this:

        for _ in range(epoch):
            model.to(device)
            for input, target in dataloader:
                input, target = input.to(device), target.to(device)

                ### YOU ARE HERE ###

                out = model(input)
                *** some magic of calculating output ***
                loss = loss_fn(output, target)

                ### THE END ###

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        the part marked with # is what this method implements
        """
        pass

    def val_on_batch(self, x: torch.Tensor, target: torch.Tensor):
        """
        Performs evaluation on a batch of data.

        Args:
            x (torch.Tensor): Input tensor.
            target (torch.Tensor): Ground truth tensor.
        """
        pass

    def predict(self, x: torch.Tensor, conf=0.6) -> torch.Tensor:
        """
        Performs prediction on an input data.

        Args:
            x (torch.Tensor): Input tensor.
            conf (float): Confidence threshold.

        Returns:
            torch.Tensor: Predicted tensor.
        """
        pass


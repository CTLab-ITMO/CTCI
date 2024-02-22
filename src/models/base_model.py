"""
an inspiration
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        """
        self.net is a backbone network
        self.cls is a classification network over the features of the net
        """

        self.net = None
        self.cls = None

    def set_loss_fn(self, loss_fn: dict):
        """
        sets loss function as a property of a model
        should be dict like {nn.Module: weight}
        """
        pass

    def forward(self, x) -> torch.tensor:
        """
        just a forward pass with all the logic implemented
        """
        pass

    def predict(self, x):
        """
        a predict method which returns a mask (for our task)
        """
        pass

    def _freeze_backbone() -> None:
        """
        this method sets self.net to eval() mode.
        might be implemented more complex, to freeze just some parts
        of the net
        """
        pass

    def _calc_loss_fn(self, input, target) -> torch.tensor:
        """
        this method calculates the loss_fn according to
        self.loss_fn. takes the module and weight from dict
        """
        pass

    def _train_on_batch(self, input, target) -> torch.tensor:
        """
        this method implements a part of training loop
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

        the part marked with # is what the this method implements
        """
        pass

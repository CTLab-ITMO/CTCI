"""
an inspiration
"""
import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, modules: torch.nn.ModuleList) -> None:
        super().__init__()
        """
        self.modules is list of model modules
        """
        self.modules = modules

    def forward(self, x) -> torch.tensor:
        """
        just a forward pass with all the logic implemented
        """
        pass

    def forward(self, x, labels) -> torch.tensor:
        """
        just a forward pass with all the logic implemented
        """
        pass

    def predict(self, x) -> torch.tensor:
        """
        a predict method which returns a mask (for our task)
        """
        pass

    def freeze_params(self) -> None:
        # TODO: how to freeze parameters
        """
        freezes part of parameters of the model
        """
        pass

    def train_on_batch(self, inputs, target) -> torch.tensor:
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

        the part marked with # is what this method implements
        """
        pass

    def val_on_batch(self, inputs, target) -> torch.tensor:
        # TODO: docstring me
        pass

    def _set_loss_fn(self, loss_fn: dict) -> None:
        """
        sets loss function as a property of a model
        should be dict like {nn.Module: weight}
        """
        pass

    def _calc_loss_fn(self, inputs, target) -> torch.tensor:
        """
        this method calculates the loss_fn according to
        self.loss_fn. takes the module and weight from dict
        """
        pass

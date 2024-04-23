import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        just a forward pass with all the logic implemented
        """
        pass

    def _calc_loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        this method calculates the loss_fn according to
        self.loss_fn. takes the module and weight from dict
        """
        pass

    def train_on_batch(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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

    def val_on_batch(self, x: torch.Tensor, target: torch.Tensor):
        """
        this method implements a part of training loop
        which is where the model gives output and then the
        validation loss and metrics are calculated
        """
        pass

    def predict(self, x: torch.Tensor, conf=0.6) -> torch.Tensor:
        """
        a predict method which returns a mask (for our task)
        """
        pass


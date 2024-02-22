import torch
from torch.optim import Adam


class Trainer():
    def __init__(self,
                 model,
                 device,
                 epoch_num,
                 optim_params: dict) -> None:

        self.model = model
        self.optim = Adam(model.parameters(), **optim_params)
        self.epoch_num = epoch_num
        self.device = device

    def train(self, train_dataloader, val_dataloader):

        for _ in range(self.epoch_num):
            # think of training strategy
            # add logger for loss
            self.model.classifier.train()
            for input, target in train_dataloader:
                input = input.to(self.device)
                target = target.to(self.device)

                self.optim.zero_grad()
                loss_train = self.model.train_on_batch(input, target)
                loss_train.backward()
                self.optim.step()

            self.model.classifier.eval()
            for input, target in val_dataloader:
                loss_val = self.model.train_on_batch(input, target)

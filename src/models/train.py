import torch
import numpy as np
from tqdm import tqdm
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
        train_epoch_loss = []
        val_epoch_loss = []
        for _ in range(self.epoch_num):

            train_batch_loss = []
            val_batch_loss = []

            # think of training strategy
            # add logger for loss
            self.model.to(self.device)
            self.model.train()
            for input, target in tqdm(train_dataloader):
                input = input.to(self.device)
                target = target.to(self.device)

                self.optim.zero_grad()
                loss_train = self.model.train_on_batch(input, target)
                train_batch_loss.append(loss_train)
                loss_train.backward()
                self.optim.step()

            self.model.eval()
            for input, target in tqdm(val_dataloader):
                input = input.to(self.device)
                target = target.to(self.device)

                loss_val = self.model.train_on_batch(input, target)
                val_batch_loss.append(loss_val)
            
            train_epoch_loss.append(np.mean(train_epoch_loss))
            val_epoch_loss.append(np.mean(val_epoch_loss))

        return train_epoch_loss, val_epoch_loss


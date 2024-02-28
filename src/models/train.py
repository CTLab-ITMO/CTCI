import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            device="cpu"
    ) -> None:

        self.model = model
        self.optim = optimizer
        self.device = device

        self.history = {"train": [], "val": []}

    def _train_epoch(self, train_dataloader):
        train_batch_loss = []
        self.model = self.model.to(self.device)
        self.model.train()
        for input, target in tqdm(train_dataloader):
            input = input.to(self.device)
            target = target.to(self.device)

            self.optim.zero_grad()
            loss_train = self.model.train_on_batch(input, target)
            train_batch_loss.append(loss_train)
            loss_train.backward()
            self.optim.step()

        return train_batch_loss

    def _val_epoch(self, val_dataloader):
        val_batch_loss = []

        self.model = self.model.to(self.device)
        self.model.eval()
        for input, target in tqdm(val_dataloader):
            input = input.to(self.device)
            target = target.to(self.device)

            loss_val = self.model.train_on_batch(input, target)
            val_batch_loss.append(loss_val)

        return val_batch_loss

    def train(self, train_dataloader, val_dataloader, epoch_num=5):
        self.history = {"train": [], "val": []}
        for epoch in range(epoch_num):
            print(f"Epoch: {epoch}")
            train_batch_loss = self._train_epoch(train_dataloader)
            val_batch_loss = self._val_epoch(val_dataloader)
            
            self.history["train"].append(np.mean(train_batch_loss))
            self.history["val"].append(np.mean(val_batch_loss))

        return self.history


class TransformerTrainer(Trainer):
    def __init__(self, model, optimizer, device="cpu"):
        super().__init__(model, optimizer, device)

    def _train_epoch(self, train_dataloader):
        train_batch_loss = []
        self.model = self.model.to(self.device)
        self.model.train()
        for batch in tqdm(train_dataloader):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optim.zero_grad()

            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            loss.backward()
            self.optim.step()

            train_batch_loss.append(loss.item())

        return train_batch_loss

    def _val_epoch(self, val_dataloader):
        val_batch_loss = []
        metric_batch = []
        self.model = self.model.to(self.device)
        self.model.eval()
        for batch in tqdm(val_dataloader):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            val_batch_loss.append(loss.item())

        return val_batch_loss

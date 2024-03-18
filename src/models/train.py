import torch
import numpy as np
from tqdm import tqdm
from src.models.utils.dirs import save_model
from src.features.adele import correct_mask, predict_average_on_scales
from src.features.adele_utils import create_labels_artifact, convert_data_to_dict, write_labels


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            metrics,
            main_metric_name="iou",
            save_dir=None,
            device="cpu"
    ) -> None:

        self.model = model
        self.optim = optimizer
        self.metrics = metrics
        self.main_metric_name = main_metric_name
        self.save_dir = save_dir
        self.device = device

        self.history = {"train": [], "val": []}
        self.metrics_num = {k: [] for k in self.metrics.keys()}

    def _train_epoch(self, train_dataloader):
        train_batch_loss = []
        self.model = self.model.to(self.device)
        self.model.train()
        for inputs, target in tqdm(train_dataloader):
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            self.optim.zero_grad()
            loss_train = self.model.train_on_batch(inputs, target)
            loss_train.backward()

            train_batch_loss.append(loss_train.item())
            self.optim.step()

        return train_batch_loss

    def _val_epoch(self, val_dataloader):
        val_batch_loss = []
        metrics_batch_num = {k: [] for k in self.metrics.keys()}

        self.model = self.model.to(self.device)
        self.model.eval()
        for inputs, target in tqdm(val_dataloader):
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            loss_val, predicted = self.model.val_on_batch(inputs, target)

            for metric_name, _ in metrics_batch_num.items():
                metric_tensor = self.metrics[metric_name](target, predicted)
                metrics_batch_num[metric_name].append(metric_tensor.item())
            val_batch_loss.append(loss_val.item())

        return val_batch_loss, metrics_batch_num
    
    def run_adele(self, tr_dataloader):
        """
        NO AUGMENTATIONS DATALOADER FOR ADELE!!!!!!!
        """
        print("running adele correction")
        create_labels_artifact()

        for images, target, names in tqdm(tr_dataloader):
            self.model.to(self.device)
            images, target = images.to(self.device), target.to(self.device)
            average = predict_average_on_scales(
                model=self.model,
                batch=images,
                scales=[0.75, 1, 1.25]
            )
            new_labels = correct_mask(
                target=target,
                average=average
            )
            data = convert_data_to_dict(names, new_labels)
            write_labels(data)

    def train(self, train_dataloader, val_dataloader, epoch_num=5, use_adele=False, adele_dataloader=None):
        self.history = {"train": [], "val": []}
        for epoch in range(epoch_num):
            print(f"\nEpoch: {epoch}")

            train_batch_loss = self._train_epoch(train_dataloader)
            val_batch_loss, metrics_batch_num = self._val_epoch(val_dataloader)

            self.history["train"].append(np.mean(train_batch_loss))
            self.history["val"].append(np.mean(val_batch_loss))

            for metric_name, metric_history in metrics_batch_num.items():
                metric_num = np.mean(metric_history)
                if (epoch_num == 1) or (metric_name == self.main_metric_name and metric_num >= max(self.metrics_num[metric_name])):
                    save_model(self.model, self.save_dir, "best.pt")
                self.metrics_num[metric_name].append(metric_num)

            if use_adele:
                self.run_adele(adele_dataloader)
                train_dataloader.dataset.use_adele = True

        save_model(self.model, self.save_dir, "last.pt")
        return self.history, self.metrics_num

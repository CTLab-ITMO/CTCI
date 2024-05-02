"""
This module implements Trainer class.

"""
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import mlflow

from src.models.base_model import BaseModel
from src.models.utils.dirs import save_model
from src.features.adele import correct_mask, predict_average_on_scales
from src.features.adele.utils import create_labels_artifact, convert_data_to_dict, write_labels


class Trainer:
    def __init__(
            self,
            model: BaseModel,
            optimizer: torch.optim.Optimizer,
            metrics: dict,
            scheduler=None,
            main_metric_name="iou",
            save_dir=None,
            device="cpu"
    ) -> None:
        """
        Initialize the Trainer class.

        Args:
            model (BaseModel): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            metrics (dict): Dictionary containing metric names as keys and corresponding metric functions as values.
            scheduler: Learning rate scheduler.
            main_metric_name (str): Name of the main metric. Default is "iou".
            save_dir (str): Directory to save the trained model. Default is None.
            device (str): Device to run the training on. Default is 'cpu'.
        """
        self.model = model
        self.optim = optimizer
        self.metrics = metrics
        self.scheduler = scheduler
        self.main_metric_name = main_metric_name
        self.save_dir = save_dir
        self.device = device

        self.history = {"train": [], "val": []}
        self.metrics_num = {k: [] for k in self.metrics.keys()}

    def _train_epoch(self, train_dataloader: torch.utils.data.DataLoader):
        """
        Train the model for one epoch.

        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.

        Returns:
            list: List containing batch-wise training losses.
        """
        train_batch_loss = []
        self.model = self.model.to(self.device)
        self.model.train()
        for inputs, target in tqdm(train_dataloader):
            inputs = inputs.to(self.device)
            target = target.to(self.device)

            self.optim.zero_grad()
            loss_train = self.model.train_on_batch(inputs, target)
            loss_train.backward()
            self.optim.step()

            mlflow.log_metric(key="train_loss", value=loss_train.item(), step=5)
            train_batch_loss.append(loss_train.item())

            # self.check_grad_norm()
            # self.check_weight_norm()

        return train_batch_loss

    def _val_epoch(self, val_dataloader: torch.utils.data.DataLoader):
        """
        Validate the model for one epoch.

        Args:
            val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

        Returns:
            tuple: Tuple containing validation losses and computed metrics.
        """
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
            mlflow.log_metric(key="val_loss", value=loss_val.item(), step=5)
            val_batch_loss.append(loss_val.item())

        return val_batch_loss, metrics_batch_num

    def run_adele(self, tr_dataloader: torch.utils.data.DataLoader) -> None:
        """
        Run Adele correction.
        NO AUGMENTATIONS DATALOADER FOR ADELE!!!!!!!

        Args:
            tr_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        """
        print("running adele correction")
        create_labels_artifact()
        with torch.no_grad():
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

    def log_intermediate_results(self, sample: torch.Tensor):
        """
        Log intermediate results during training.

        Args:
            sample (torch.Tensor): Sample input data.
        """
        # TODO: do it at least beautiful
        transform = transforms.ToPILImage()
        with torch.no_grad():
            self.model.eval()
            self.model.to("cpu")
            out = self.model.predict(sample)
            sample = transform(sample[0])
            out = transform(torch.tensor(out[0]*255, dtype=torch.uint8))
            mlflow.log_image(image=sample, artifact_file="input.png")
            mlflow.log_image(image=out, artifact_file="output.png")

    def check_grad_norm(self) -> float:
        """
        Check gradient norm during training.

        Returns:
            float: Total gradient norm.
        """
        total_norm = 0
        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        mlflow.log_metric(key="grad_norm", value=total_norm, step=5)
        return total_norm

    def check_weight_norm(self) -> float:
        """
        Check weight norm during training.

        Returns:
            float: Total weight norm.
        """
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        mlflow.log_metric(key="weight_norm", value=total_norm, step=5)

        return total_norm

    def train(
            self,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader,
            epoch_num=5, use_adele=False, adele_dataloader=None
    ):
        """
        Train the model.

        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            epoch_num (int): Number of epochs to train. Default is 5.
            use_adele (bool): Whether to use Adele correction. Default is False.
            adele_dataloader (torch.utils.data.DataLoader): DataLoader for Adele correction. Default is None.

        Returns:
            tuple: Tuple containing training history and computed metrics.
        """
        self.history = {"train": [], "val": []}
        for epoch in range(epoch_num):
            print(f"\nEpoch: {epoch}")

            train_batch_loss = self._train_epoch(train_dataloader)
            val_batch_loss, metrics_batch_num = self._val_epoch(val_dataloader)

            if self.scheduler:
                self.scheduler.step()

            self.log_intermediate_results(next(iter(train_dataloader))[0])

            print(np.mean(train_batch_loss))
            self.history["train"].append(np.mean(train_batch_loss))
            self.history["val"].append(np.mean(val_batch_loss))

            for metric_name, metric_history in metrics_batch_num.items():
                metric_num = np.mean(metric_history)
                if (epoch == 0) or (metric_name == self.main_metric_name and metric_num >= max(self.metrics_num[metric_name])):
                    save_model(self.model, self.save_dir, "best.pt")
                self.metrics_num[metric_name].append(metric_num)

            if use_adele:
                self.run_adele(adele_dataloader)
                train_dataloader.dataset.use_adele = True

        save_model(self.model, self.save_dir, "last.pt")
        return self.history, self.metrics_num

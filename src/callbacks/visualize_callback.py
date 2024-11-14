import lightning.pytorch as pl
import torch
import numpy as np
import matplotlib.pyplot as plt


class VisualizationCallback(pl.Callback):
    def __init__(self, log_interval=1, max_images=4, alpha=0.5):
        """
        Args:
            log_interval (int): Number of epochs between visualizations.
            max_images (int): Number of images to visualize from a batch.
            alpha (float): Transparency for overlaying the mask on the image.
        """
        super().__init__()
        self.log_interval = log_interval
        self.max_images = max_images
        self.alpha = alpha

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_interval != 0:
            return

        # Get a batch of validation data
        val_dataloader = trainer.datamodule.val_dataloader()
        val_batch = next(iter(val_dataloader))
        images = val_batch[0]

        images = images.to(pl_module.device)

        # Predict masks
        with torch.no_grad():
            predictions = pl_module(images)
            predictions = torch.sigmoid(predictions)
            predictions = predictions > 0.5

        # Visualize the first few images
        fig, axes = plt.subplots(self.max_images, 3, figsize=(15, 5 * self.max_images))
        for i in range(min(self.max_images, len(images))):
            image = images[i].cpu().permute(1, 2, 0).numpy()
            image = (image - image.min()) / (image.max() - image.min())  # Normalize for display

            pred_mask = predictions[i].cpu().squeeze().numpy()  # Squeeze for single channel
            overlay = self._overlay_mask(image, pred_mask)

            # Original Image
            axes[i, 0].imshow(image)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis("off")

            # Prediction
            axes[i, 1].imshow(pred_mask, cmap="Blues")
            axes[i, 1].set_title("Prediction")
            axes[i, 1].axis("off")

            # Overlay
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title("Overlay")
            axes[i, 2].axis("off")

        plt.tight_layout()

        # Log figure to TensorBoard or other logger
        trainer.logger.experiment.add_figure("Segmentation Results", fig, trainer.current_epoch)
        plt.close(fig)

    def _overlay_mask(self, image, mask):
        """Helper function to overlay a binary mask on an image."""
        overlay = image.copy()
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)  # Ensure mask has 3 channels
        overlay = np.clip(overlay * (1 - self.alpha) + mask * self.alpha, 0, 1)
        return overlay

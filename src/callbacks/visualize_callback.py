import lightning.pytorch as pl
import torch
import matplotlib.pyplot as plt


class VisualizationCallback(pl.Callback):
    def __init__(self, max_images=4, alpha=0.5):
        """
        Args:
            max_images (int): Number of images to visualize from a batch.
            alpha (float): Transparency for overlaying the mask on the image.
        """
        super().__init__()
        self.max_images = max_images
        self.alpha = alpha

    def on_validation_epoch_end(self, trainer, pl_module):
        logger = trainer.logger
        if not hasattr(logger, "experiment") or not logger.experiment:
            print("ClearML logger not found or not initialized.")
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

        # Visualize and log results
        fig, axes = plt.subplots(self.max_images, 3, figsize=(15, 5 * self.max_images))
        for i in range(min(self.max_images, len(images))):
            image = images[i].cpu().permute(1, 2, 0).numpy()
            image = (image - image.min()) / (image.max() - image.min())  # Normalize for display
            pred_mask = predictions[i].cpu().numpy()
            overlay = self._overlay_mask(image, pred_mask)

            # Plot original, prediction, and overlay
            axes[i, 0].imshow(image)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(pred_mask, cmap="Blues")
            axes[i, 1].set_title("Prediction")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title("Overlay")
            axes[i, 2].axis("off")

        plt.tight_layout()

        # Log figure to ClearML
        logger.experiment.report_matplotlib_figure(
            title="Segmentation Results",
            series="Validation Results",
            figure=fig,
            iteration=trainer.current_epoch,
        )
        plt.close(fig)

    def _overlay_mask(self, image, mask):
        if mask.ndim == 2:
            mask = mask[:, :, None]
        elif mask.shape[0] == 1:
            mask = mask.squeeze(0)[:, :, None]

        mask = mask.repeat(3, axis=-1)
        overlay = (image * (1 - self.alpha) + mask * self.alpha).clip(0, 1)
        return overlay


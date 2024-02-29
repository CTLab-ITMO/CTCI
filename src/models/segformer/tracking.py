import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import transformers
from torch.utils.data import DataLoader

import mlflow

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.train import Trainer
from src.models.segformer.model import SegFormer
from src.models.metrics import IoUMetric


def draw_results(model):
    image = Image.open(r"C:\Internship\ITMO_ML\CTCI\data\frame-0.png")
    predicted_segmentation_map = model.predict(image)
    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

    ax[0].imshow(image)
    ax[1].imshow(predicted_segmentation_map, cmap="gray")
    plt.show()
    return fig


if __name__ == "__main__":
    train_images_dir = sys.argv[1]
    train_masks_dir = sys.argv[2]
    val_images_dir = sys.argv[3]
    val_masks_dir = sys.argv[4]
    save_path = sys.argv[5]

    train_batch_size = 6
    val_batch_size = 4
    pin_memory = True
    num_workers = 4

    image_processor = transformers.SegformerImageProcessor(size={"width": 256, "height": 256})

    train_dataset = SegmentationDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        image_processor=image_processor
    )

    val_dataset = SegmentationDataset(
        images_dir=val_images_dir,
        masks_dir=val_masks_dir,
        image_processor=image_processor
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        pin_memory=pin_memory, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=val_batch_size,
        pin_memory=pin_memory, num_workers=num_workers
    )

    net = transformers.SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-ade-512-512"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    model = SegFormer(net=net, image_processor=image_processor, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    metric = IoUMetric().to(device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        metric=metric,
        device=device
    )

    mlflow.set_experiment(experiment_name="segformer")
    mlflow.autolog()
    with mlflow.start_run(run_name="segformer-b1-test_metrics"):
        history = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epoch_num=3
        )

        mlflow.set_tag('model', 'SegFormer-b1')
        mlflow.set_tag('device', 'cuda')
        mlflow.set_tag('augmentation', 'None')
        mlflow.set_tag('train_dataset', 'Weakly segmented ..\\test')
        mlflow.set_tag('val_dataset', 'Weakly segmented ..\\val')

        fig = draw_results(model)
        mlflow.log_figure(fig, 'segformer-b1_results.png')
        mlflow.log_metric("iou", np.mean(history["metric"]))
        mlflow.end_run()

    torch.save(model.state_dict(), save_path)

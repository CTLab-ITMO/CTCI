import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt

import torch

import mlflow
from torch.utils.data import DataLoader

from src.models.metrics import Recall, Precision, Accuracy, DiceMetric, IoUMetric
from src.models.train import Trainer


def draw_results(model):
    image = Image.open(r"C:\Internship\ITMO_ML\CTCI\data\test_data\bubbles\frame-4.png")
    predicted_segmentation_map = model.predict(image)
    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

    ax[0].imshow(image)
    ax[1].imshow(predicted_segmentation_map, cmap="gray")
    plt.show()
    return fig


def tracking_run(
        trainer,
        train_dataloader, val_dataloader,
        config_data, run_name=None
):
    train_batch_size = config_data['dataloader']['train_batch_size']
    val_batch_size = config_data['dataloader']['val_batch_size']

    model_name = config_data['model']['model_name']
    model_type = config_data['model']['model_type']
    model_save_dir = config_data['model']['save_dir']

    adele = config_data['training']['adele']
    epoch_num = config_data['training']['epoch_num']

    optimizer_name = config_data['optimizer']['name']
    optimizer_lr = config_data['optimizer']['lr']

    draw_plots = config_data['history']['draw_plots']
    draw_result = config_data['history']['draw_result']
    plots_figure_name = config_data['history']['plots_figure_name']
    results_figure_name = config_data['history']['results_figure_name']

    if run_name is None:
        run_name = f"{model_name}-{model_type}"

    mlflow.autolog()
    with mlflow.start_run(run_name=run_name):
        history, metrics_num = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epoch_num=epoch_num
        )

        mlflow.set_tag("model", f"{model_name}-{model_type}")
        mlflow.set_tag("optimizer_name", optimizer_name)

        mlflow.log_param("train_batch_size", train_batch_size)
        mlflow.log_param("val_batch_size", val_batch_size)
        mlflow.log_param("adele", adele)
        mlflow.log_param("epoch_num", epoch_num)
        mlflow.log_param("lr", optimizer_lr)

        if draw_result:
            results_fig = draw_results(trainer.model)
            mlflow.log_figure(results_fig, results_figure_name)
            plt.savefig(osp.join(model_save_dir, results_figure_name))

        for metric_name, metric_history in metrics_num.items():
            mlflow.log_metric(metric_name, metric_history[-1])

        if draw_plots:
            pass  # TODO: draw, log and save metrics and loss plots

        mlflow.end_run()

    torch.save(trainer.model.state_dict(), osp.join(model_save_dir, "last.pt"))


def tracking_experiment(
        model,
        train_dataset, val_dataset,
        config_data,
        experiment_name="experiment"
):
    # TODO: возобновляемость экспериментов
    random_state = config_data['random_state']

    image_size = config_data['dataset']['image_size']

    train_batch_size = config_data['dataloader']['train_batch_size']
    val_batch_size = config_data['dataloader']['val_batch_size']
    pin_memory = config_data['dataloader']['pin_memory']
    num_workers = config_data['dataloader']['num_workers']

    device_name = config_data['model']['device']

    image_processor = model.image_processor
    if image_processor:
        image_processor.size = image_size
        train_dataset.image_processor = image_processor
        val_dataset.image_processor = image_processor

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        pin_memory=pin_memory, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=val_batch_size,
        pin_memory=pin_memory, num_workers=num_workers
    )

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if device_name.split(':')[0] == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True

    model.device = device

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = {
        "iou": IoUMetric().to(device),
        "dice": DiceMetric().to(device),
        "accuracy": Accuracy().to(device),
        "precision": Precision().to(device),
        "recall": Recall().to(device)
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        device=device
    )

    mlflow.set_experiment(experiment_name=experiment_name)
    tracking_run(
        trainer=trainer,
        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
        config_data=config_data,
        run_name=None
    )

import os
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt

import torch

import mlflow
from torch.utils.data import DataLoader

from src.models.utils.dirs import check_dir, create_folder, create_run_folder
from src.models.utils.reproducibility import set_seed
from src.models.metrics import Recall, Precision, Accuracy, DiceMetric, IoUMetric
from src.models.train import Trainer


# TODO: refactor me
def draw_results(model, show_plot=False):
    images_dir = '.\\data\\test_data\\bubbles'
    images_list = os.listdir(images_dir)
    figs = {image_name: [] for image_name in images_list}
    for image_name in images_list:
        image = Image.open(os.path.join(images_dir, image_name))
        predicted_segmentation_map = model.predict(image)
        predicted_segmentation_map = predicted_segmentation_map.squeeze().cpu().detach().numpy()
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

        ax[0].imshow(image)
        ax[1].imshow(predicted_segmentation_map, cmap="gray")
        figs[image_name] = fig
        if show_plot:
            plt.show()
    return figs


# TODO: refactor me
def draw_history(history, metrics_num, show_plot=False, width=8, fontsize=20):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

    ax[0].plot(
        range(len(history['train'])), history['train'],
        label="train", linewidth=width
    )
    ax[0].plot(
        range(len(history['val'])), history['val'], 'r--',
        label="val", linewidth=width
    )
    ax[0].set_xlabel("epochs")
    ax[0].set_title("history")
    ax[0].tick_params(axis='both', which='major')
    ax[0].tick_params(axis='both', which='minor')
    ax[0].grid(True)
    ax[0].legend(fontsize=fontsize)

    for i, (metric_name, metric_value) in enumerate(metrics_num.items()):
        ax[1].plot(
            range(len(metric_value)), metric_value,
            label=metric_name, linewidth=width
        )
    ax[1].set_xlabel("epochs")
    ax[1].set_title("metrics")
    ax[1].tick_params(axis='both', which='major')
    ax[1].tick_params(axis='both', which='minor')
    ax[1].grid(True)
    ax[1].legend(fontsize=fontsize)
    if show_plot:
        plt.show()
    return fig


def tracking_run(
        trainer,
        train_dataloader, val_dataloader,
        config_data, run_name=None
):
    random_seed = config_data['random_seed']

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
    plots_line_width = config_data['history']['plots_line_width']
    plots_fontsize = config_data['history']['plots_fontsize']
    plots_figure_name = config_data['history']['plots_figure_name']
    results_figure_name = config_data['history']['results_figure_name']

    if run_name == 'None':
        run_name = f"{model_name}-{model_type}"
    if model_save_dir == 'None':
        model_save_dir = trainer.save_dir

    mlflow.autolog()
    with mlflow.start_run(run_name=run_name):
        history, metrics_num = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epoch_num=epoch_num,
            use_adele=adele
        )

        mlflow.set_tag("model", f"{model_name}-{model_type}")
        mlflow.set_tag("optimizer_name", optimizer_name)

        mlflow.log_param("train_batch_size", train_batch_size)
        mlflow.log_param("val_batch_size", val_batch_size)
        mlflow.log_param("adele", adele)
        mlflow.log_param("epoch_num", epoch_num)
        mlflow.log_param("lr", optimizer_lr)
        mlflow.log_param("random_seed", random_seed)

        if draw_result:
            results_figs = draw_results(trainer.model)
            for image_name, results_fig in results_figs.items():
                mlflow.log_figure(results_fig, f"{image_name}_{results_figure_name}")
                results_fig.savefig(osp.join(model_save_dir, f"{image_name}_{results_figure_name}"))

        if draw_plots:
            history_fig = draw_history(
                history=history, metrics_num=metrics_num,
                width=plots_line_width, fontsize=plots_fontsize
            )
            mlflow.log_figure(history_fig, plots_figure_name)
            history_fig.savefig(osp.join(model_save_dir, plots_figure_name))

        for metric_name, metric_history in metrics_num.items():
            mlflow.log_metric(metric_name, metric_history[-1])

        mlflow.end_run()


def tracking_experiment(
        model,
        train_dataset, val_dataset,
        config_data,
        experiment_name="experiment"
):
    random_seed = config_data['random_seed']
    set_seed(random_seed)

    image_size = config_data['dataset']['image_size']

    train_batch_size = config_data['dataloader']['train_batch_size']
    val_batch_size = config_data['dataloader']['val_batch_size']
    pin_memory = config_data['dataloader']['pin_memory']
    num_workers = config_data['dataloader']['num_workers']

    device_name = config_data['model']['device']

    if hasattr(model, 'image_processor'):
        image_processor = model.image_processor
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

    device = torch.device(device_name)
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
    main_metric_name = config_data["training"]["main_metric"]
    if main_metric_name not in [str(metric) for metric in metrics]:
        main_metric_name = "iou"

    model_save_dir = config_data['model']['save_dir']
    if model_save_dir == 'None':
        directory = f".\\checkpoints\\{experiment_name}"
        if not check_dir(directory):
            create_folder(directory)
        model_save_dir = create_run_folder(directory)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        main_metric_name=main_metric_name,
        save_dir=model_save_dir,
        device=device
    )

    mlflow.set_experiment(experiment_name=experiment_name)
    config_data["mlflow"]["experiment_name"] = experiment_name
    tracking_run(
        trainer=trainer,
        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
        config_data=config_data
    )

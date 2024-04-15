import os.path as osp
import pandas as pd

import torch

import mlflow
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from src.models.utils.dirs import check_dir, create_folder, create_run_folder
from src.models.utils.reproducibility import set_seed
from src.models.utils.models_settings import set_image_processor_to_datasets, set_gpu
from src.models.utils.config import ConfigHandler
from src.models.metrics import Recall, Precision, Accuracy, DiceMetric, IoUMetric
from src.models.train import Trainer
from src.visualization.visualization import draw_results, draw_history


def tracking_run(
        trainer,
        train_dataloader, val_dataloader, adele_dataloader,
        config_handler: ConfigHandler, run_name=None
):
    random_seed = config_handler.read('random_seed')

    train_batch_size = config_handler.read('dataloader', 'train_batch_size')
    val_batch_size = config_handler.read('dataloader', 'val_batch_size')

    model_name = config_handler.read('model', 'model_name')
    model_type = config_handler.read('model', 'model_type')
    model_save_dir = config_handler.read('model', 'save_dir')

    optimizer_name = config_handler.read('optimizer', 'name')
    optimizer_lr = config_handler.read('optimizer', 'lr')

    adele = config_handler.read('training', 'adele')
    epoch_num = config_handler.read('training', 'epoch_num')

    draw_plots = config_handler.read('history', 'draw_plots')
    draw_result = config_handler.read('history', 'draw_result')
    plots_line_width = config_handler.read('history', 'plots_line_width')
    plots_fontsize = config_handler.read('history', 'plots_fontsize')
    plots_figure_name = config_handler.read('history', 'plots_figure_name')
    results_figure_name = config_handler.read('history', 'results_figure_name')

    if run_name == 'None':
        run_name = f"{model_name}-{model_type}"
    if model_save_dir == 'None':
        model_save_dir = trainer.save_dir

    mlflow.autolog()
    with mlflow.start_run(run_name=run_name):
        history, metrics_num = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            adele_dataloader=adele_dataloader,
            epoch_num=epoch_num,
            use_adele=adele
        )

        mlflow.set_tag('model', f"{model_name}-{model_type}")
        mlflow.set_tag('optimizer_name', optimizer_name)

        mlflow.log_param('train_batch_size', train_batch_size)
        mlflow.log_param('val_batch_size', val_batch_size)
        mlflow.log_param('adele', adele)
        mlflow.log_param('epoch_num', epoch_num)
        mlflow.log_param('lr', optimizer_lr)
        mlflow.log_param('random_seed', random_seed)

        report_df = pd.DataFrame(metrics_num)
        report_df['train_loss'] = history['train']
        report_df['val_loss'] = history['val']
        report_df.to_csv(osp.join(model_save_dir, f"report.csv"))
        mlflow.log_artifact(osp.join(model_save_dir, f"report.csv"))

        mlflow.log_metric('train_loss', history['train'][-1])
        mlflow.log_metric('val_loss', history['val'][-1])

        for metric_name, metric_history in metrics_num.items():
            mlflow.log_metric(metric_name, metric_history[-1])

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

        mlflow.end_run()


def tracking_experiment(
        model,
        train_dataset, val_dataset,
        config_handler: ConfigHandler,
        scheduler=None,
        adele_dataset=None,
        experiment_name="experiment"
):
    random_seed = config_handler.read('random_seed')
    set_seed(random_seed)
    adele_dataloader = None  # TODO: fix adele

    image_size = config_handler.read('dataset', 'image_size')

    train_batch_size = config_handler.read('dataloader', 'train_batch_size')
    val_batch_size = config_handler.read('dataloader', 'val_batch_size')
    pin_memory = config_handler.read('dataloader', 'pin_memory')
    num_workers = config_handler.read('dataloader', 'num_workers')

    device_name = config_handler.read('model', 'device')

    set_image_processor_to_datasets(model, image_size, [train_dataset, val_dataset])
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        pin_memory=pin_memory, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=val_batch_size,
        pin_memory=pin_memory, num_workers=num_workers
    )
    if adele_dataset:
        adele_dataloader = DataLoader(
            adele_dataset, batch_size=train_batch_size,
            pin_memory=pin_memory, num_workers=num_workers
        )

    device = set_gpu(device_name)
    model.device = device

    optimizer_lr = config_handler.read('optimizer', 'lr')
    betas = config_handler.read('optimizer', 'betas')
    weight_decay = config_handler.read('optimizer', 'weight_decay')

    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_lr, betas=betas, weight_decay=weight_decay)

    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            lr_scheduler.LinearLR(optimizer),
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000),
        ],
        milestones=[2]
    )
    # TODO: fix scheduler
    # scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=8, power=0.9)

    metrics = {
        "iou": IoUMetric().to(device),
        "dice": DiceMetric().to(device),
        "accuracy": Accuracy().to(device),
        "precision": Precision().to(device),
        "recall": Recall().to(device)
    }
    main_metric_name = config_handler.read("training", "main_metric")
    if main_metric_name not in [str(metric) for metric in metrics]:
        main_metric_name = "iou"

    model_save_dir = config_handler.read('model', 'save_dir')
    if model_save_dir == 'None':
        directory = f".\\checkpoints\\{experiment_name}"
        if not check_dir(directory):
            create_folder(directory)
        model_save_dir = create_run_folder(directory)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        main_metric_name=main_metric_name,
        save_dir=model_save_dir,
        device=device
    )

    mlflow.set_experiment(experiment_name=experiment_name)
    config_handler.config_data["mlflow"]["experiment_name"] = experiment_name
    tracking_run(
        trainer=trainer,
        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
        adele_dataloader=adele_dataloader,
        config_handler=config_handler
    )

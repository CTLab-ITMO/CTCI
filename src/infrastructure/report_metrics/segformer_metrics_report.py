import sys
import os.path as osp

from torch.utils.data import DataLoader

from src.features.segmentation.dataset import SegmentationDataset
from src.models.segformer.segformer import build_segformer
from src.models.metrics import Recall, Precision, Accuracy, DiceMetric, IoUMetric, ReportMetrics
from src.models.utils.config import read_yaml_config
from src.models.utils.models_settings import set_image_processor_to_datasets, set_gpu


# TODO: create metrics report function for
if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    model_name = config_handler.read('model', 'model_name')
    model_type = config_handler.read('model', 'model_type')

    model = build_segformer(config_handler)

    test_dataset = SegmentationDataset(
        images_dir=osp.join(config_handler.read('dataset', 'test_dataset_dirs')[0], "images"),
        masks_dir=osp.join(config_handler.read('dataset', 'test_dataset_dirs')[0], "masks"),
    )

    image_size = config_handler.read('dataset', 'image_size')

    batch_size = config_handler.read('dataloader', 'train_batch_size')
    pin_memory = config_handler.read('dataloader', 'pin_memory')
    num_workers = config_handler.read('dataloader', 'num_workers')

    device_name = config_handler.read('model', 'device')

    set_image_processor_to_datasets(model, image_size, [test_dataset])

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=pin_memory, num_workers=num_workers
    )

    device = set_gpu(device_name)
    model.device = device

    metrics = {
        "iou": IoUMetric().to(device),
        "dice": DiceMetric().to(device),
        "accuracy": Accuracy().to(device),
        "precision": Precision().to(device),
        "recall": Recall().to(device)
    }

    report_metrics = ReportMetrics(
        model=model,
        metrics=metrics,
        device=device
    )

    metrics_num = report_metrics.run_metrics(test_dataloader=test_dataloader)

    print(metrics_num)

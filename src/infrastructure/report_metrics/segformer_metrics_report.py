import sys
import os.path as osp

from torch.utils.data import DataLoader
import transformers

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.segformer.model import SegFormer
from src.models.metrics import Recall, Precision, Accuracy, DiceMetric, IoUMetric, ReportMetrics
from src.models.utils.config import read_yaml_config
from src.models.utils.models_settings import set_image_processor_to_datasets, set_gpu


# TODO: create metrics report function for
if __name__ == "__main__":
    config_path = sys.argv[1]
    config_data = read_yaml_config(config_path)

    model_name = config_data['model']['model_name']
    model_type = config_data['model']['model_type']

    image_processor = transformers.SegformerImageProcessor()
    net = transformers.SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/{model_name}-{model_type}-finetuned-ade-512-512"
    )
    model = SegFormer(net=net, image_processor=image_processor, device=config_data['model']['device'])

    test_dataset = SegmentationDataset(
        images_dir=osp.join(config_data['dataset']['test_dataset_dirs'][0], "images"),
        masks_dir=osp.join(config_data['dataset']['test_dataset_dirs'][0], "masks"),
        image_processor=image_processor
    )

    image_size = config_data['dataset']['image_size']

    batch_size = config_data['dataloader']['train_batch_size']
    pin_memory = config_data['dataloader']['pin_memory']
    num_workers = config_data['dataloader']['num_workers']

    device_name = config_data['model']['device']

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

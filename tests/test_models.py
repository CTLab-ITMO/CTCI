import os.path as osp

import pytest
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.segformer.model import SegFormer
from src.models.hrnet.model import HRNet


@pytest.fixture(scope="package")
def segformer_model():
    net = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b1-finetuned-ade-512-512")
    image_processor = SegformerImageProcessor(),
    device = 'cpu'
    model = SegFormer(net, image_processor, device)
    return model


@pytest.fixture(scope="package")
def hrnet_model():
    model = HRNet(False)
    return model


@pytest.fixture(scope="package")
def train_data_path():
    path = '..\\data\\weakly_segmented\\bubbles_split\\train'
    return path


@pytest.fixture(scope="package")
def val_data_path():
    path = '..\\data\\weakly_segmented\\bubbles_split\\valid'
    return path


@pytest.fixture
def train_dataset(train_data_path):
    train_dataset = SegmentationDataset(
        images_dir=osp.join(train_data_path, "images"),
        masks_dir=osp.join(train_data_path, "masks")
    )
    return train_dataset


@pytest.fixture
def val_dataset(val_data_path):
    val_dataset = SegmentationDataset(
        images_dir=osp.join(val_data_path, "images"),
        masks_dir=osp.join(val_data_path, "masks")
    )
    return val_dataset


class TestModels:
    @pytest.mark.parametrize(
        ['model_class', 'args'],
        [
            (SegFormer, (
                SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b1-finetuned-ade-512-512"),
                SegformerImageProcessor(),
                'cpu'
            )),
            (SegFormer, (
                SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b1-finetuned-ade-512-512"),
                None,
                'cpu'
            )),
            # TODO: HRNet cases
            (HRNet, (
                True,
            )),
            (HRNet, (
                False,
            )),
        ]
    )
    def test_model_init(self, model_class, args):
        model = model_class(*args)
        assert isinstance(model, model_class)

    @pytest.mark.skip(reason="Not implemented")
    def test_model_predict(self):
        pass


class TestTraining:
    # TODO: сделать skip по флагу в CLI
    @pytest.mark.skip(reason="There is no cuda on Mac :(")
    def test_device(self):
        assert torch.cuda.is_available()

    @pytest.mark.parametrize(
        ["model"],
        [
            (segformer_model, ),
            (hrnet_model, )
        ]
    )
    def test_model_train_on_batch(self, model):
        # TODO: дописать
        pass

    @pytest.mark.skip(reason="Not implemented")
    def test_model_val_on_batch(self):
        pass

    @pytest.mark.skip(reason="Not implemented")
    def test_model_epoch(self):
        pass

    @pytest.mark.skip(reason="Not implemented")
    def test_model_adele(self):
        pass


class TestTracking:
    @pytest.mark.skip(reason="Not implemented")
    def test_tracking_run(self):
        pass

    @pytest.mark.skip(reason="Not implemented")
    def test_tracking_experiment(self):
        pass



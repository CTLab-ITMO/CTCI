import os.path as osp

import pytest
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.segformer.model import SegFormer
from src.models.hrnet.model import HRNet


class TestTraining:
    # TODO: сделать skip по флагу в CLI
    @pytest.mark.skip(reason="There is no cuda on Mac :(")
    def test_device(self):
        assert torch.cuda.is_available()

    @pytest.mark.parametrize(
        ["model"],
        [
            # TODO: add models and datasets as cases
        ],
        indirect=True
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


import os.path as osp

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from src.features.segmentation.dataset import SegmentationDataset
from src.models.segformer import SegFormer
from src.models.hrnet import HRNetModel


class TestTransformerTraining:
    # TODO: сделать skip по флагу в CLI не подходящих или долгих тестов

    def test_device(self):
        assert torch.cuda.is_available()

    @pytest.mark.parametrize(
        ['model_name', 'dataset_name'],
        [
            ('segformer_model', 'train_dataset'),
        ]
    )
    def test_transformer_model_train_on_batch(self, model_name, dataset_name, request):
        model = request.getfixturevalue(model_name)
        dataset = request.getfixturevalue(dataset_name)

        dataloader = DataLoader(
            dataset, batch_size=2, shuffle=True,
            pin_memory=False, num_workers=0
        )

        for inputs, targets in dataloader:
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)
            res = model.train_on_batch(inputs, targets)

            assert res is not None
            break

    @pytest.mark.parametrize(
        ['model_name', 'dataset_name'],
        [
            ('segformer_model', 'val_dataset'),
        ]
    )
    def test_transformer_model_val_on_batch(self, model_name, dataset_name, request):
        model = request.getfixturevalue(model_name)
        dataset = request.getfixturevalue(dataset_name)

        dataloader = DataLoader(
            dataset, batch_size=2, shuffle=False,
            pin_memory=False, num_workers=0
        )

        for inputs, targets in dataloader:
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)
            res = model.val_on_batch(inputs, targets)

            assert res is not None
            break

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


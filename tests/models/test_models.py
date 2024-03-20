import os.path as osp

import pytest
import torch
from transformers import SegformerForSemanticSegmentation, Swinv2Model, SegformerImageProcessor, AutoImageProcessor

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.segformer.model import SegFormer
from src.models.hrnet.model import HRNet
from src.models.swin.model import Swin


class TestModels:
    @pytest.mark.skip(reason="Not implemented")
    def test_model_predict(self):
        pass

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
            (Swin, (
                Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256"),
                AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256"),
                'cpu'
            )),
            (Swin, (
                Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256"),
                None,
                'cpu'
            )),
            (Swin, (
                Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256"),
                AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256"),
                'cuda'
            )),
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

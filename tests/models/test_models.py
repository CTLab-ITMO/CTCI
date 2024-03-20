import os.path as osp

import pytest
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.segformer.model import SegFormer
from src.models.hrnet.model import HRNet


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
            # (HRNet, (
            #    True,
            # )),
            # (HRNet, (
            #     False,
            # )),
        ]
    )
    def test_model_init(self, model_class, args):
        model = model_class(*args)
        assert isinstance(model, model_class)

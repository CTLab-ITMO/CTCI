import os.path as osp
from PIL import Image

import pytest
import torch
from transformers import SegformerForSemanticSegmentation, Swinv2Model, SegformerImageProcessor, AutoImageProcessor

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.segformer.model import SegFormer
from src.models.hrnet.model import HRNet
from src.models.swin.model import Swin


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

    @pytest.mark.parametrize(
        ['model_name', 'image_path'],
        [
            ('segformer_model', '.\\data\\test_data\\bubbles\\frame-0.png'),
            ('swin_model', '.\\data\\test_data\\bubbles\\frame-0.png')
        ]
    )
    def test_model_predict(self, model_name, image_path, request):
        model = request.getfixturevalue(model_name)
        device = 'cuda'
        model.device = device
        model = model.to(device)
        image = Image.open(image_path)

        predicted_seg_map = model.predict(image)
        assert predicted_seg_map is not None

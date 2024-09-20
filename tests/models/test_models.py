import os.path as osp
from PIL import Image

import timm
import pytest

import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation, Swinv2Model

from src.models.segformer import SegFormer
from src.models.hrnet import HRNetModel
from src.models.swin.model import Swin


class TestModels:
    @pytest.mark.parametrize(
        ['model_class', 'args'],
        [
            (SegFormer, (
                SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b1-finetuned-ade-512-512"),
                None,
                None,
                (256, 256),
                'cpu'
            )),
            (SegFormer, (
                SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b1-finetuned-ade-512-512"),
                None,
                None,
                (256, 256),
                'cpu'
            )),
            (Swin, (
                Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256"),
                None,
                None,
                (256, 256),
                'cpu'
            )),
            (Swin, (
                Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256"),
                None,
                None,
                (256, 256),
                'cpu'
            )),
            (Swin, (
                Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window16-256"),
                None,
                None,
                (256, 256),
                'cuda'
            )),
            (HRNetModel, (
                timm.create_model('hrnet_w18_small_v2', features_only=True, pretrained=True),
                None,
                None,
                (256, 256),
                'cpu'
            )),
            (HRNetModel, (
                timm.create_model('hrnet_w18_small_v2', features_only=True, pretrained=True),
                None,
                None,
                (256, 256),
                'cuda'
            )),
        ]
    )
    def test_model_init(self, model_class, args):
        model = model_class(*args)
        assert isinstance(model, model_class)

    @pytest.mark.parametrize(
        ['model_name', 'image_path'],
        [
            ('segformer_model', './data/test_data/bubbles/frame-0.png'),
            ('swin_model', './data/test_data/bubbles/frame-0.png')
        ]
    )
    def test_model_predict(self, model_name, image_path, request):
        to_tensor = transforms.ToTensor()

        model = request.getfixturevalue(model_name)
        device = 'cuda'
        model.device = device
        model = model.to(device)

        image = Image.open(image_path)
        image = to_tensor(image).unsqueeze(0).to(device)

        predicted_seg_map = model.predict(image)
        assert predicted_seg_map is not None

import os.path as osp

import pytest
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers import Swinv2Model, AutoImageProcessor

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.segformer.model import SegFormer
from src.models.hrnet.model import HRNet
from src.models.swin.model import Swin


@pytest.fixture(scope='function')
def segformer_model():
    net = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b1-finetuned-ade-512-512")
    image_processor = SegformerImageProcessor()
    device = 'cpu'
    model = SegFormer(net=net, image_processor=image_processor, device=device)
    return model


@pytest.fixture(scope='function')
def hrnet_model():
    model = HRNet(freeze_backbone=False)
    return model


@pytest.fixture(scope='function')
def swin_model():
    model_str = rf"microsoft/swinv2-tiny-patch4-window16-256"
    image_processor = AutoImageProcessor.from_pretrained(model_str)
    net = Swinv2Model.from_pretrained(model_str)
    model = Swin(net=net, image_processor=image_processor, device='cpu')
    return model

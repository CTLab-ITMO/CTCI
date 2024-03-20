import os.path as osp

import pytest
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.segformer.model import SegFormer
from src.models.hrnet.model import HRNet


@pytest.fixture(scope='function')
def segformer_model():
    net = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b1-finetuned-ade-512-512")
    image_processor = SegformerImageProcessor(),
    device = 'cpu'
    model = SegFormer(net, image_processor, device)
    return model


@pytest.fixture(scope='function')
def hrnet_model():
    model = HRNet(False)
    return model


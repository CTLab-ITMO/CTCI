import pytest

from src.models.segformer.model import build_segformer
from src.models.hrnet.model import build_hrnet
from src.models.swin.model import build_swin
from src.models.utils.config import read_yaml_config


@pytest.fixture(scope='function')
def segformer_model():
    config_handler = read_yaml_config('./src/infrastructure/configs/training/training_segformer_config.yaml')
    model = build_segformer(config_handler)
    return model


@pytest.fixture(scope='function')
def hrnet_model():
    config_handler = read_yaml_config('./src/infrastructure/configs/training/training_hrnet_config.yaml')
    model = build_hrnet(config_handler)
    return model


@pytest.fixture(scope='function')
def swin_model():
    config_handler = read_yaml_config('./src/infrastructure/configs/training/training_swin_config.yaml')
    model = build_swin(config_handler)
    return model

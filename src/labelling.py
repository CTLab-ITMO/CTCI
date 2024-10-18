from argparse import ArgumentParser
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from src.transform import rename_state_dict_keys


def init_model(config_path='../../configs', config_name='config', arch=None):
    with initialize(version_base=None, config_path=config_path):
        overrides = []
        if arch:
            overrides.append("module.arch={}".format(arch))
        # Compose the config, applying the overrides
        cfg = compose(config_name=config_name, overrides=overrides)
        model = instantiate(cfg.module.arch)
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="../configs")
    parser.add_argument('--config_name', type=str, default="config")
    parser.add_argument('--arch', type=str, default="deeplab3")
    parser.add_argument("--rename_keys", action='store_true')
    parser.add_argument('--weights', type=str)

    args = parser.parse_args()

    model = init_model(config_path=args.config_path, config_name=args.config_name)
    state_dict = torch.load(args.weights)['state_dict']
    if args.rename_keys:
        state_dict = rename_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)

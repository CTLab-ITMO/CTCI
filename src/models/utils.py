import os
import os.path as osp

import torch


def check_dir(directory):
    return osp.exists(directory)


def create_folder(directory):
    os.mkdir(directory)


def save_model(model, directory, weights_name):
    if not check_dir(directory):
        create_folder(directory)
    torch.save(model.state_dict(), osp.join(directory, weights_name))

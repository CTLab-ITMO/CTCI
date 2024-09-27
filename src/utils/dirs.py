"""
This module provides utility functions for working with directories.

"""
import os
import os.path as osp

import torch

from src.models.base_model import BaseModel


def check_dir(directory: str) -> bool:
    """
    Check if a directory exists.

    Args:
        directory (str): Path to the directory.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    return osp.exists(directory)


def create_folder(directory: str) -> None:
    """
    Create a folder if it doesn't exist.

    Args:
        directory (str): Path to the directory to be created.
    """
    os.mkdir(directory)


def save_model(model: BaseModel, directory: str, weights_name: str) -> None:
    """
    Save a PyTorch model to a directory.

    Args:
        model (torch.nn.Module): PyTorch model to be saved.
        directory (str): Path to the directory where the model will be saved.
        weights_name (str): Name of the file to save the model weights.
    """
    if not check_dir(directory):
        create_folder(directory)
    torch.save(model.state_dict(), osp.join(directory, weights_name))


def determine_run_folder(directory: str) -> str:
    """
    Determine the name of the next available run folder.

    Args:
        directory (str): Path to the directory containing run folders.

    Returns:
        str: Name of the next available run folder.
    """
    folder_name = "run"
    run_num = 1
    folders_list = os.listdir(directory)
    find_folder = False
    while True:
        if f"{folder_name}_{run_num}" in folders_list:
            if len(os.listdir(osp.join(directory, f"{folder_name}_{run_num}"))) != 0:
                run_num += 1
            else:
                find_folder = True
        else:
            break
        if find_folder:
            break

    folder_name = f"{folder_name}_{run_num}"
    return folder_name


def create_run_folder(directory: str) -> str:
    """
    Create a new run folder.

    Args:
        directory (str): Path to the directory where the run folder will be created.

    Returns:
        str: Path to the newly created run folder.
    """
    folder_name = determine_run_folder(directory)
    directory = osp.join(directory, folder_name)
    if not check_dir(directory):
        create_folder(directory)
    return directory

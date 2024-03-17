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


def determine_run_folder(directory):
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


def create_run_folder(directory):
    folder_name = determine_run_folder(directory)
    directory = osp.join(directory, folder_name)
    if not check_dir(directory):
        create_folder(directory)
    return directory

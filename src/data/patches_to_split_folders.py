import os
import os.path as osp
import shutil
import json

with open("./src/data/data_config.json", 'r') as f:
    d = json.load(f)
    train_folders = d['TRAIN_FOLDERS']
    test_folders = d['TEST_FOLDERS']
    valid_folders = d['VALID_FOLDERS']
    output_folder = d['OUTPUT_FOLDER']


splits = {
    #"train": train_folders,
    "test": test_folders,
    "valid": valid_folders
}

for output_folder_name, folders in splits.items():
    for folder in folders:
        folder_name = osp.basename(osp.normpath(folder))
        for f in os.listdir(folder):
            shutil.copyfile(
                osp.join(folder,f),
                osp.join(output_folder, 
                         output_folder_name,
                         folder_name + "_" + f)
            )
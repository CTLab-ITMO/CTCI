import os
import os.path as osp
from sklearn.model_selection import train_test_split
import shutil

FOLDERS_DIR = r""
RANDOM_STATE = 239


def get_folders_list(folders_dir: str):
    return os.listdir(folders_dir)


def copy_files(folder_name, set):
    if not osp.exists(os.path.join(FOLDERS_DIR, folder_name)):
        os.mkdir(os.path.join(FOLDERS_DIR, folder_name))

    for image in set:
        image_name = image.split("\\")[-2] + "_" + image.split("\\")[-1]
        shutil.copy(src=os.path.join(FOLDERS_DIR, image), dst=os.path.join(FOLDERS_DIR, folder_name, image_name))


if __name__ == "__main__":
    folders_list = get_folders_list(FOLDERS_DIR)

    images_list = []

    for folder in folders_list:
        folder_images_list = os.listdir(os.path.join(FOLDERS_DIR, folder))

        for image in folder_images_list:
            images_list.append(os.path.join(folder, image))

    train, val = train_test_split(images_list, train_size=0.9, random_state=RANDOM_STATE)

    copy_files(folder_name="train", set=train)
    copy_files(folder_name="val", set=val)


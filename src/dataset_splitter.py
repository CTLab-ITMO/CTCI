import os
import random
import shutil
from typing import List
from sklearn.model_selection import train_test_split
from src.utils.files_utils import clean_hidden_files


def split_data(
    src_folder: str,
    dst_folder: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> None:
    random.seed(seed)

    image_folder = src_folder
    mask_folder = f"{src_folder}_masks"

    if dst_folder is None:
        dst_folder = src_folder

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"Image folder '{image_folder}' does not exist.")

    masks_exist = os.path.isdir(mask_folder)
    if not masks_exist:
        print("You should provide masks or perform annotation.")
        return

    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)

    image_files = clean_hidden_files(image_files)
    mask_files = clean_hidden_files(mask_files)
    assert len(image_files) == len(mask_files), "The number of images and masks must be equal."

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dst_folder, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dst_folder, split, "masks"), exist_ok=True)

    train_files, val_files, test_files = splitter(
        image_files,
        train_ratio,
        val_ratio,
        test_ratio,
        seed,
    )

    # Move files to respective folders
    move_files(
        train_files,
        split="train",
        image_folder=image_folder,
        mask_folder=mask_folder,
        dst_folder=dst_folder,
    )
    move_files(
        val_files,
        split="val",
        image_folder=image_folder,
        mask_folder=mask_folder,
        dst_folder=dst_folder,
    )
    move_files(
        test_files,
        split="test",
        image_folder=image_folder,
        mask_folder=mask_folder,
        dst_folder=dst_folder,
    )
    os.removedirs(mask_folder)

    print(f"Data successfully split into train ({train_ratio}), val ({val_ratio}), and test ({test_ratio}).")


def splitter(files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):

    train_files, temp_files = train_test_split(files, train_size=train_ratio, random_state=seed)
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(temp_files, train_size=val_test_ratio, random_state=seed)

    return train_files, val_files, test_files


def move_files(
        file_list: List[str],
        split: str,
        image_folder: str,
        mask_folder: str,
        dst_folder: str,
) -> None:
    for file_name in file_list:
        src_image = os.path.join(image_folder, file_name)
        dest_image = os.path.join(dst_folder, split, "images", file_name)
        shutil.move(src_image, dest_image)

        src_mask = os.path.join(mask_folder, file_name)
        dest_mask = os.path.join(dst_folder, split, "masks", file_name)
        if os.path.exists(src_mask):
            shutil.move(src_mask, dest_mask)

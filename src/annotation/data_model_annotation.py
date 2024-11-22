import os
import cv2
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision.transforms import Resize
from src.transform import cv_image_to_tensor, tensor_to_cv_image
from src.utils.model_utils import load_model
from src.utils.files_utils import clean_hidden_files
from src.constants import PROJECT_ROOT
from src.logger import LOGGER


@hydra.main(version_base=None, config_path='../../configs', config_name='model_annotation')
def run_annotation(cfg: DictConfig) -> None:
    LOGGER.info(f'Annotating folder {cfg.folder}')
    source_dir = os.path.join(PROJECT_ROOT, cfg.folder)
    output_dir = os.path.join(PROJECT_ROOT, cfg.folder + "_masks")

    model = hydra.utils.instantiate(cfg.module)
    model.load_state_dict(load_model(cfg.pretrained_path))
    LOGGER.info(f'Loaded model from {cfg.pretrained_path}')
    conf = cfg.conf
    resize = Resize(cfg.data.img_size)

    for image_name in tqdm(clean_hidden_files(os.listdir(source_dir))):
        image_path = os.path.join(source_dir, image_name)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = cv_image_to_tensor(image).unsqueeze(0)  # add batch
        image = resize(image)

        mask = model.predict(image, conf=conf)
        mask = tensor_to_cv_image(mask.squeeze(0))

        mask_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_mask.png")
        cv2.imwrite(mask_path, mask)
    LOGGER.info(f'Finished annotating. Results saved to {output_dir}')


if __name__ == '__main__':
    run_annotation()

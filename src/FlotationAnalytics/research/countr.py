!git clone https://github.com/Verg-Avesta/CounTR.git

!pip install -r /content/CounTR/requirements.txt --upgrade --force-reinstall

!pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
!pip install timm==0.4.9 #0.3.2
!pip install numpy
!pip install matplotlib tqdm
!pip install tensorboard
!pip install scipy
!pip install imgaug
!pip install opencv-python
!pip3 install hub

"""
Изменить в файле /content/CounTR/util/misc.py from torch._six import inf на inf = float('inf')
Изменить в файле /content/CounTR/util/pos_embed.py omega = np.arange(embed_dim // 2, dtype=np.float) на omega = np.arange(embed_dim // 2, dtype=float)
"""

with open('/usr/local/lib/python3.11/dist-packages/timm/models/layers/helpers.py', 'r') as f:
  text = f.read()

print(text)

!python /content/CounTR/demo_zero.py --input_path frames/ --output_path results/

"""заменить файл demo_zero.py"""
import cv2
import os
import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
from itertools import chain
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms

import timm
import models_mae_cross
from util.misc import measure_time

assert "0.4.5" <= timm.__version__ <= "0.4.9"  # Проверяем версию timm
device = torch.device('cuda')
shot_num = 0

def extract_frames(video_path, output_folder):
    """
    Разбиваем видео на кадры и сохраняем их в output_folder.
    Также вычисляем средний вектор перемещения объектов.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception(f"Ошибка: Не удалось открыть видео {video_path}")

    frame_idx = 0
    prev_gray = None
    total_motion = np.array([0.0, 0.0])
    num_vectors = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Определяем средний вектор движения объектов
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mean_flow = np.mean(flow, axis=(0, 1))
            total_motion += mean_flow
            num_vectors += 1

        prev_gray = frame_gray
        frame_idx += 1

    cap.release()

    avg_motion = total_motion / num_vectors if num_vectors > 0 else np.array([0, 0])
    print(f"Кадры сохранены в {output_folder} ({frame_idx} кадров).")
    print(f"Средний вектор перемещения объектов: ({avg_motion[0]:.2f}, {avg_motion[1]:.2f})")
    return frame_idx, avg_motion

def frames_to_video(input_folder, output_video, fps=30):
    """Преобразуем кадры обратно в видео"""
    frame_files = sorted(Path(input_folder).glob("viz_*.jpg"))

    if not frame_files:
        raise Exception("Ошибка: Не найдено обработанных кадров!")

    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        out.write(frame)

    out.release()
    print(f"Видео сохранено: {output_video}")

def load_image(img_path: str):
    image = Image.open(img_path).convert('RGB')
    image.load()
    W, H = image.size
    new_H = 384
    new_W = 16 * int((W / H * 384) / 16)
    image = transforms.Resize((new_H, new_W))(image)
    Normalize = transforms.Compose([transforms.ToTensor()])
    image = Normalize(image)
    boxes = torch.Tensor([])
    return image, boxes, W, H

def run_one_image(samples, boxes, model, output_path, img_name, old_w, old_h):
    _, _, h, w = samples.shape

    density_map = torch.zeros([h, w])
    density_map = density_map.to(device, non_blocking=True)
    start = 0
    prev = -1
    with measure_time() as et:
        with torch.no_grad():
            while start + 383 < w:
                output, = model(samples[:, :, :, start:start + 384], boxes, shot_num)
                output = output.squeeze(0)
                b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                d1 = b1(output[:, 0:prev - start + 1])
                b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                d2 = b2(output[:, prev - start + 1:384])

                b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                density_map_l = b3(density_map[:, 0:start])
                density_map_m = b1(density_map[:, start:prev + 1])
                b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                density_map_r = b4(density_map[:, prev + 1:w])

                density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                prev = start + 383
                start = start + 128
                if start + 383 >= w:
                    if start == w - 384 + 128:
                        break
                    else:
                        start = w - 384

        pred_cnt = torch.sum(density_map / 60).item()

    fig = samples[0]
    pred_fig = torch.stack((density_map, torch.zeros_like(density_map), torch.zeros_like(density_map)))
    count_im = Image.new(mode="RGB", size=(w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(count_im)
    draw.text((w-70, h-50), f"{pred_cnt:.3f}", (255, 255, 255))
    count_im = np.array(count_im).transpose((2, 0, 1))
    count_im = torch.tensor(count_im, device=device)
    fig = fig / 2 + pred_fig / 2 + count_im
    fig = torch.clamp(fig, 0, 1)
    fig = transforms.Resize((old_h, old_w))(fig)
    torchvision.utils.save_image(fig, output_path / f'viz_{img_name}.jpg')
    return pred_cnt, et

if __name__ == '__main__':
    start_time = time.time()

    video_path = "/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/PseCo/video_cut1.mp4"
    frames_folder = "frames"
    output_video = "output_video.mp4"

    # разбиваем видео на кадры и аналирируем движение
    num_frames, avg_motion = extract_frames(video_path, frames_folder)

    p = ArgumentParser()
    p.add_argument("--input_path", type=Path, required=True)
    p.add_argument("--output_path", type=Path, default="results")
    p.add_argument("--model_path", type=Path, default="/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/CounTR/FSC147.pth")

    args = p.parse_args()

    args.output_path.mkdir(exist_ok=True, parents=True)

    model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu')['model'], strict=False)
    model.eval()

    if args.input_path.is_dir():
        inputs = sorted(list(chain(args.input_path.glob("*.jpg"), args.input_path.glob("*.png"))))
        for i, img_path in enumerate(inputs):
            samples, boxes, old_w, old_h = load_image(img_path)
            result, elapsed_time = run_one_image(samples.unsqueeze(0).to(device), boxes.unsqueeze(0).to(device), model, args.output_path, img_path.stem, old_w, old_h)
            print(f"[{i+1}/{len(inputs)}] {img_path.name}:\tcount = {result:.2f}  -  time = {elapsed_time.duration:.2f}")

    frames_to_video("results", output_video)

    print(f"Время работы программы: {time.time() - start_time:.2f} секунд")
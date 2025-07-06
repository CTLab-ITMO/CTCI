### Загружаем YOLO
!pip install opencv-python-headless torch torchvision torchaudio
!git clone https://github.com/ultralytics/yolov5
!wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt

### По сегментации делаем разметку
import cv2
import os
import numpy as np

# Извлечение bbox из маски
def extract_bounding_boxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x+w, y+h])
    return boxes

# Конвертация bbox в YOLO формат
def convert_bbox_to_yolo(box, image_width, image_height):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return x_center, y_center, width, height

def process_images_to_yolo_labels(image_folder, mask_folder, output_folder, class_id=0):
    os.makedirs(output_folder, exist_ok=True)

    for image_name in os.listdir(image_folder):
        if not image_name.lower().endswith('.jpg'):
            continue

        base_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, f"{base_name}.png")
        txt_path = os.path.join(output_folder, f"{base_name}.txt")

        if not os.path.exists(mask_path):
            print(f"Маска для {image_name} не найдена. Пропуск...")
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Ошибка при загрузке изображения: {image_path}")
            continue
        if mask is None:
            print(f"Ошибка при загрузке маски: {mask_path}")
            continue

        boxes = extract_bounding_boxes(mask)
        if not boxes:
            print(f"Не найдено объектов на изображении {image_name}")
            continue

        height, width = image.shape[:2]
        with open(txt_path, 'w') as f:
            for box in boxes:
                yolo_coords = convert_bbox_to_yolo(box, width, height)
                f.write(f"{class_id} {' '.join(f'{x:.6f}' for x in yolo_coords)}\n")

        print(f"Разметка для {image_name} сохранена в {txt_path}")

if __name__ == "__main__":
    image_folder = '/content/drive/MyDrive/Проекты/Мультитрекинг однородных объектов/YOLOv5-QCB+SORT/data/val/images'
    mask_folder = '/content/drive/MyDrive/Проекты/Мультитрекинг однородных объектов/YOLOv5-QCB+SORT/data/val/masks'
    output_folder = '/content/drive/MyDrive/Проекты/Отслеживание в реальном времени/Sort + other methods time/labels'

    process_images_to_yolo_labels(image_folder, mask_folder, output_folder)


### Обучаем YOLO

data_yaml = """
train: /content/drive/MyDrive/Проекты/Мультитрекинг однородных объектов/YOLOv5-QCB+SORT/data/train/images
val: /content/drive/MyDrive/Проекты/Мультитрекинг однородных объектов/YOLOv5-QCB+SORT/data/val/images

nc: 1
names: ['bubble']
"""

data_path = '/content/data.yaml'

with open(data_path, 'w') as f:
    f.write(data_yaml)

import os
import subprocess

data_yaml = "data.yaml"
model_weights = "yolov5s.pt"
epochs = 50
batch_size = 10
conf_thres = 0.1
max_det = 1000

command = f"python ./yolov5/train.py --data {data_yaml} --cfg yolov5s.yaml --weights {model_weights} --epochs {epochs} --batch-size {batch_size} --hyp hyp.scratch.yaml --img 640 --device 0 --conf-thres {conf_thres} --max-det {max_det}"
subprocess.run(command, shell=True)

import time
import os

start_time = time.time()

!yolo train data=data.yaml model=yolov5s.pt epochs=50 batch=10 conf=0.1 max_det=1000

end_time = time.time()
print(f"Время тренировки: {end_time - start_time:.2f} секунд")

### Детекция с помощью YOLO

import time
import cv2
import os
from ultralytics import YOLO

model = YOLO('/content/runs/detect/train/weights/best.pt')

video_path = '/content/drive/MyDrive/Проекты/Мультитрекинг однородных объектов/YOLOv5-QCB+SORT/input_yolov5.mp4'
detections_folder = '/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/Видео/детекции_пузыри_3/'
os.makedirs(detections_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Ошибка при открытии видео")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = '/content/drive/MyDrive/Проекты/Мультитрекинг однородных объектов/YOLOv5-QCB+SORT/out_yolov5.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Обработка кадров
frame_number = 0
total_time = 0
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Применяем детекцию объектов на текущем кадре
    results = model(frame, max_det=1000)

    end_time = time.time()

    inference_time = end_time - start_time
    total_time += inference_time
    frame_count += 1

    print(f"Кадр {frame_number}: время инференса = {inference_time:.4f} сек")

    # Получаем аннотации для текущего кадра (формат: x1, y1, x2, y2, conf, class)
    annotations = results[0].boxes.data.cpu().numpy()

    # Проверка на наличие детекций
    if annotations.size > 0:
        # Создаем текстовый файл для текущего кадра, если есть детекции
        detections_path = os.path.join(detections_folder, f"frame_{frame_number:05d}.txt")
        with open(detections_path, "w") as f:
            for box in annotations:
                x1, y1, x2, y2, conf, class_id = box
                width = x2 - x1
                height = y2 - y1
                f.write(f"{int(class_id)},{x1},{y1},{width},{height},{conf}\n")
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    out.write(frame)
    frame_number += 1

cap.release()
out.release()

# Выводим среднее время инференса
print(f"\nОбщее время: {total_time:.4f} сек")
if frame_count > 0:
    avg_time = total_time / frame_count
    print(f"\nСреднее время инференса: {avg_time:.4f} сек/кадр\n")
    print(f"FPS инференса: {1 / avg_time:.2f} кадров/сек\n")
else:
    print("Не удалось обработать ни одного кадра.")
print(f"Детекции для каждого кадра сохранены в {detections_folder}")

### Устанавливаем все необходимое, чтобы открыть расширение .mkv

!apt-get update
!apt-get install -y ffmpeg

!ffmpeg -i "/content/42.ФМ ь 23-001.mkv"

# !ffmpeg -i "/content/42.ФМ ь 23-001.mkv" -ss 00:00:00 -to 00:00:08 -c copy "/content/new_video1.mkv" #обрезка видео

!ffmpeg -i "/content/42.ФМ ь 23-001.mkv" -ss 00:00:00 -to 00:00:08 -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k "/content/video_cut1.mp4"

### Установка SORT

!git clone https://github.com/abewley/sort.git

!pip install filterpy==1.4.5
!pip install scikit-image==0.18.1

### Установка lap

!python -m pip install --upgrade pip

!pip install --upgrade lap

!git clone https://github.com/gatagat/lap.git

import lap

### Делим видео на кадры

import cv2
import os

def split_video_into_frames(video_path, frames_folder):
    os.makedirs(frames_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_folder, f"frame_{frame_number:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_number += 1

    cap.release()
    print(f"Видео разделено на {frame_number} кадров и сохранено в папке {frames_folder}")

split_video_into_frames('/content/drive/MyDrive/Проекты/Мультитрекинг однородных объектов/YOLOv5-QCB+SORT/out_yolov5.mp4', '/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/Видео/frame_detection')

!pip install filterpy

!python /content/sort.py

### Наносим трекинг на изначальную видеозапись
import os
import cv2
number_of_frames = 100
def visualize_tracking(video_path, tracked_data_folder, output_video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    for frame_number in range(number_of_frames):
        ret, frame = cap.read()
        if not ret:
            break

        tracked_path = os.path.join(tracked_data_folder, f"tracked_{frame_number:05d}.txt")
        if os.path.exists(tracked_path):
            with open(tracked_path, "r") as f:
                for line in f:
                    x1, y1, x2, y2, obj_id = map(float, line.strip().split(","))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {int(obj_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

visualize_tracking('/content/drive/MyDrive/Проекты/Мультитрекинг однородных объектов/YOLOv5-QCB+SORT/input_yolov5.mp4','/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/finish','/content/last_video_cut1.mp4')

# заменить код sort.py
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import cv2

np.random.seed(0)

def temporal_consistency(bubble_centers):
    consistency_scores = []
    for t in range(len(bubble_centers) - 1):
        frame1 = bubble_centers[t]
        frame2 = bubble_centers[t + 1]

        if len(frame1) == len(frame2):
            distances = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(frame1, frame2)]
            consistency_scores.append(np.mean(distances))
        else:
            continue

    if consistency_scores:
        return np.mean(consistency_scores)
    else:
        print("No valid consistency scores calculated. Returning 0.")
        return 0

def optical_flow_similarity(prev_frame, next_frame, prev_centers, next_centers):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    similarities = []
    for (x1, y1), (x2, y2) in zip(prev_centers, next_centers):
        try:
            dx, dy = flow[int(y1), int(x1)]
            motion_vector = np.array([x2 - x1, y2 - y1])
            flow_vector = np.array([dx, dy])

            if np.linalg.norm(motion_vector) > 0 and np.linalg.norm(flow_vector) > 0:
                cos_sim = np.dot(motion_vector, flow_vector) / (np.linalg.norm(motion_vector) * np.linalg.norm(flow_vector))
                similarities.append(cos_sim)
        except Exception as e:
            print(f"Error calculating optical flow for centers ({x1}, {y1}) and ({x2}, {y2}): {e}")
            continue

    return np.mean(similarities) if similarities else None

def object_recall_watershed(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    frame_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(frame_colored, markers)
    frame_colored[markers == -1] = [0, 0, 255]
    return np.max(markers) - 1


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)

  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
  return(o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    if w <= 0 or h <= 0:
        print(f"Invalid bounding box: {bbox}")
        return np.array([0, 0, 0, 0]).reshape((4, 1))

    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000.  #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

def analyze_tracking(pred_tracks):
    total_objects = 0
    total_tracks = 0
    track_durations = []
    prev_frame_objects = set()
    new_objects = 0
    lost_objects = 0

    for frame_index in sorted(pred_tracks.keys()):
        frame = pred_tracks[frame_index]
        current_frame_objects = set(frame)
        total_objects += len(frame)

        # Считаем количество новых объектов (FP) и потерянных объектов (FN)
        new_objects += len(current_frame_objects - prev_frame_objects)
        lost_objects += len(prev_frame_objects - current_frame_objects)

        # Считаем продолжительность треков (каждый трек - это объект, который появляется на нескольких кадрах)
        for obj in frame:
            track_durations.append(frame_index + 1)  # Длительность - это номер кадра, на котором появился объект

        prev_frame_objects = current_frame_objects
        total_tracks += len(frame)

    # Среднее количество объектов на кадр
    avg_objects_per_frame = total_objects / len(pred_tracks) if len(pred_tracks) > 0 else 0
    # Средняя продолжительность треков
    avg_track_duration = np.mean(track_durations) if len(track_durations) > 0 else 0
    # Частота появления новых объектов (FP) и исчезновения объектов (FN)
    new_object_frequency = new_objects / len(pred_tracks) if len(pred_tracks) > 0 else 0
    lost_object_frequency = lost_objects / len(pred_tracks) if len(pred_tracks) > 0 else 0

    return {
        "avg_objects_per_frame": avg_objects_per_frame,
        "avg_track_duration": avg_track_duration,
        "new_object_frequency": new_object_frequency,
        "lost_object_frequency": lost_object_frequency
    }

if __name__ == "__main__":
    frames_folder = '/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/Видео/frame_detection'
    detections_folder = '/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/Видео/детекции_пузыри_3/'
    output_folder = '/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/finish/'
    os.makedirs(output_folder, exist_ok=True)

    tracker = Sort()
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])

    bubble_centers = {}

    for frame_number, frame_file in enumerate(frame_files):
        detections_path = os.path.join(detections_folder, f"frame_{frame_number:05d}.txt")
        detections = []

        with open(detections_path, "r") as f:
            for line in f:
                det = list(map(float, line.strip().split(",")))
                x1, y1, width, height, conf = det[1:]
                x2, y2 = x1 + width, y1 + height
                detections.append([x1, y1, x2, y2, conf])

        detections = np.array(detections) if detections else np.empty((0, 5))
        tracked_objects = tracker.update(detections)

        bubble_centers[frame_number] = []
        if len(tracked_objects) > 0:
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = obj
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                bubble_centers[frame_number].append((center_x, center_y))
            with open(os.path.join(output_folder, f"tracked_{frame_number:05d}.txt"), "w") as f:
                for obj in tracked_objects:
                    f.write(",".join(map(str, obj)) + "\n")

        current_frame = cv2.imread(os.path.join(frames_folder, frame_file))

        if current_frame is None:
            print(f"Error: Could not load frame {frame_number}")
            continue

        watershed_bubble_count = object_recall_watershed(current_frame)

        # Анализ темпоральной согласованности и оптического потока
        if frame_number > 0:
            prev_frame = cv2.imread(os.path.join(frames_folder, frame_files[frame_number - 1]))

            if prev_frame is None:
                print(f"Error: Could not load previous frame {frame_files[frame_number - 1]}")
                continue

            temporal_score = temporal_consistency(bubble_centers)
            flow_score = optical_flow_similarity(prev_frame, current_frame,
                                                 bubble_centers[frame_number - 1],
                                                 bubble_centers[frame_number])

            print(f"Frame {frame_number}: Temporal Consistency = {temporal_score}, "
                  f"Optical Flow Similarity = {flow_score}, "
                  f"Watershed Bubble Count = {watershed_bubble_count}")

    analysis = analyze_tracking(bubble_centers)

    print(f"Среднее количество объектов на кадр: {analysis['avg_objects_per_frame']:.2f}")
    print(f"Средняя продолжительность треков: {analysis['avg_track_duration']:.2f} кадров")
    print(f"Частота появления новых объектов (FP): {analysis['new_object_frequency']:.2f}")
    print(f"Частота исчезновения объектов (FN): {analysis['lost_object_frequency']:.2f}")
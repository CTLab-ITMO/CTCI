import cv2
import numpy as np

video_path = "/content/drive/MyDrive/Проекты/Мультитрекинг однородных объектов/YOLOv5-QCB+SORT/input_yolov5.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: не удалось открыть видео.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Параметры детектора углов (Shi-Tomasi)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Параметры Лукаса-Канаде для оптического потока
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
if not ret:
    print("Ошибка при загрузке первого кадра")
    cap.release()
    out.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)

    else:
        img = frame

    out.write(img)
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

cap.release()
out.release()

print(f"Видео сохранено как {output_path}")

import cv2
import numpy as np
import time

video_path = "/content/drive/MyDrive/Проекты/Мультитрекинг однородных объектов/YOLOv5-QCB+SORT/input_yolov5.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: не удалось открыть видео.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "output_tracking.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

ret, old_frame = cap.read()
if not ret:
    print("Ошибка при загрузке первого кадра")
    cap.release()
    out.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

start_time = time.time()

# Хранение среднего вектора перемещения по всем кадрам
total_dx, total_dy, frame_count = 0, 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Вычисляем оптический поток (метод Лукаса-Канаде)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)

    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        dx_values, dy_values = [], []

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            dx, dy = a - c, b - d
            dx_values.append(dx)
            dy_values.append(dy)
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        if dx_values and dy_values:
            avg_dx = np.mean(dx_values)
            avg_dy = np.mean(dy_values)
        else:
            avg_dx, avg_dy = 0, 0

        total_dx += avg_dx
        total_dy += avg_dy
        frame_count += 1

        text = f"Avg Flow: ({avg_dx:.2f}, {avg_dy:.2f})"
        cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        p0 = good_new.reshape(-1, 1, 2)

    output_frame = cv2.add(frame, mask)
    out.write(output_frame)
    old_gray = frame_gray.copy()

global_avg_dx = total_dx / frame_count if frame_count > 0 else 0
global_avg_dy = total_dy / frame_count if frame_count > 0 else 0

end_time = time.time()
total_time = end_time - start_time

cap.release()
out.release()

print(f"Видео сохранено как {output_path}")
print(f"Время работы мультитрекинга: {total_time:.2f} секунд")
print(f"Средний вектор перемещения пузырей за всё видео: ({global_avg_dx:.2f}, {global_avg_dy:.2f})")

import cv2
import numpy as np
import time

video_path = "/content/drive/MyDrive/Проекты/Отслеживание_в_реальном_времени/Sort+other_methods_time/Optical_flow/video_cut1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: не удалось открыть видео.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "output_плотный.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
ret, old_frame = cap.read()
if not ret:
    print("Ошибка при загрузке первого кадра")
    cap.release()
    out.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

start_time = time.time()
total_dx, total_dy, frame_count = 0, 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Вычисляем плотный оптический поток (метод Гуннара Фарнебака)
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    step = 15
    dx_values, dy_values = [], []

    for y in range(0, frame.shape[0], step):
        for x in range(0, frame.shape[1], step):
            fx, fy = flow[y, x]
            end_x = int(x + fx)
            end_y = int(y + fy)

            if np.sqrt(fx**2 + fy**2) > 2:
                cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 0, 255), 1, tipLength=0.3)
                dx_values.append(fx)
                dy_values.append(fy)

    if dx_values and dy_values:
        avg_dx = np.mean(dx_values)
        avg_dy = np.mean(dy_values)
    else:
        avg_dx, avg_dy = 0, 0

    total_dx += avg_dx
    total_dy += avg_dy
    frame_count += 1

    text = f"Avg Flow: ({avg_dx:.2f}, {avg_dy:.2f})"
    cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    out.write(frame)
    old_gray = frame_gray.copy()

global_avg_dx = total_dx / frame_count if frame_count > 0 else 0
global_avg_dy = total_dy / frame_count if frame_count > 0 else 0

end_time = time.time()
total_time = end_time - start_time
cap.release()
out.release()

print(f"Видео сохранено как {output_path}")
print(f"Время работы мультитрекинга: {total_time:.2f} секунд")
print(f"Средний вектор перемещения пузырей за всё видео: ({global_avg_dx:.2f}, {global_avg_dy:.2f})")
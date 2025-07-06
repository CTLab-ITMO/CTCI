# Подсчет контролируемых метрик: MOTP и MOTA
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import shutil


# Переименовываем файлы с image0.txt в image{i}.txt
def rename_yolo_annotations(root_dir):
    root_path = Path(root_dir)
    for labels_dir in root_path.glob("track*/labels"):
        track_num = labels_dir.parent.name.replace("track", "")
        if track_num == "":
            track_num = 0
        else:
            track_num = int(track_num) - 1

        for txt_file in labels_dir.glob("image0.txt"):
            new_name = f"image{track_num}.txt"
            new_path = txt_file.with_name(new_name)
            shutil.move(str(txt_file), str(new_path))
            print(f"Renamed: {txt_file} -> {new_path}")


# Парсим разметки из текстовых файлов и возвращаем словарь с разметкой
# Для CVAT аннотации
def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find("meta/original_size/width").text)
    height = int(root.find("meta/original_size/height").text)

    annotations = {}

    for track in root.findall("track"):
        track_id = int(track.get("id"))

        for box in track.findall("box"):
            frame = int(box.get("frame"))
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            box_width = xbr - xtl
            box_height = ybr - ytl
            x_center = xtl + box_width / 2
            y_center = ytl + box_height / 2

            if width and height:
                x_center /= width
                y_center /= height
                box_width /= width
                box_height /= height

                x_center = max(0, min(x_center, 1.0))
                y_center = max(0, min(y_center, 1.0))
                box_width = max(0, min(box_width, 1.0))
                box_height = max(0, min(box_height, 1.0))

            if frame not in annotations:
                annotations[frame] = []

            annotations[frame].append(
                {
                    "track_id": int(track_id) + 1,
                    "bbox": [x_center, y_center, box_width, box_height],
                }
            )

    return annotations


# Для расширенной YOLO аннотации
def parse_yolo_txt(yolo_dir):
    annotations = {}

    yolo_dir = Path(yolo_dir)
    labels_dirs = list(yolo_dir.glob("*/labels"))

    for labels_dir in labels_dirs:
        for txt_file in labels_dir.glob("*.txt"):
            match = re.search(r"image(\d+).txt", txt_file.name)
            if not match:
                continue

            frame = int(match.group(1))

            with open(txt_file, "r") as f:
                lines = f.readlines()

            if frame not in annotations:
                annotations[frame] = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    conf = float(parts[5])
                    track_id = float(parts[6])

                    annotations[frame].append(
                        {
                            "track_id": int(track_id),
                            "bbox": [x_center, y_center, width, height],
                            "confidence": conf,
                        }
                    )
    return annotations


def calculate_iou(bbox1, bbox2):
    try:
        x1, y1, w1, h1 = map(float, bbox1)
        x2, y2, w2, h2 = map(float, bbox2)
    except (TypeError, ValueError) as e:
        print(f"Ошибка формата bbox: {e}")
        return 0.0

    w1, h1 = max(0, w1), max(0, h1)
    w2, h2 = max(0, w2), max(0, h2)

    inter_x1 = max(x1 - w1 / 2, x2 - w2 / 2)
    inter_y1 = max(y1 - h1 / 2, y2 - h2 / 2)
    inter_x2 = min(x1 + w1 / 2, x2 + w2 / 2)
    inter_y2 = min(y1 + h1 / 2, y2 + h2 / 2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1 * h1 + w2 * h2 - inter_area + 1e-6
    return inter_area / union_area


# Оптимальное сопоставление с использованием венгерского алгоритма
def hungarian_matching(cvat_boxes, yolo_boxes, iou_threshold):
    cost_matrix = 1 - np.array(
        [[calculate_iou(c["bbox"], y["bbox"]) for y in yolo_boxes] for c in cvat_boxes]
    )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if 1 - cost_matrix[r, c] >= iou_threshold:
            matches.append((r, c))

    return matches


# Сопоставление ограничивающих рамок
def hybrid_matcher(cvat_boxes, yolo_boxes, iou_threshold, dist_threshold=0):
    # Венгерский алгоритм (сопоставление по IoU)
    matches = hungarian_matching(cvat_boxes, yolo_boxes, iou_threshold)

    # Сопоставление оставшихся ограничивающих рамок, которые не нашли пары на предыдущем шаге, по расстоянию
    matched_cvat = set(m[0] for m in matches)
    matched_yolo = set(m[1] for m in matches)

    for i, c_box in enumerate(cvat_boxes):
        if i in matched_cvat:
            continue

        min_dist = dist_threshold
        best_j = -1
        c_pos = np.array(c_box["bbox"][:2])

        for j, y_box in enumerate(yolo_boxes):
            if j in matched_yolo:
                continue

            y_pos = np.array(y_box["bbox"][:2])
            dist = np.linalg.norm(c_pos - y_pos)

            if dist < min_dist:
                min_dist = dist
                best_j = j

        if best_j != -1:
            matches.append((i, best_j))
            matched_yolo.add(best_j)

    unmatched_cvat = [i for i in range(len(cvat_boxes)) if i not in matched_cvat]
    unmatched_yolo = [j for j in range(len(yolo_boxes)) if j not in matched_yolo]

    return matches, unmatched_cvat, unmatched_yolo


# Фильтрация ложных срабатываний
def filter_bboxes(bboxes, min_size=0, min_conf=0):
    return [
        b
        for b in bboxes
        if (b["bbox"][2] * b["bbox"][3] >= min_size)
        and b.get("confidence", 1.0) >= min_conf
    ]


def draw_comparison(
    cvat_boxes, yolo_boxes, matches, img_size=(1058, 793), filename="comparison.jpg"
):
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

    # Цвета в формате BGR (для OpenCV)
    COLOR_GT = (0, 200, 0)
    COLOR_YOLO = (0, 100, 255)
    COLOR_MATCH = (255, 255, 0)
    COLOR_BG = (251, 251, 251)

    # Заливаем фон
    img[:] = COLOR_BG

    for box in cvat_boxes:
        x, y, w, h = (np.array(box["bbox"]) * img_size[1]).astype(int)
        cv2.rectangle(
            img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 0, 0), 3
        )
        cv2.rectangle(
            img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), COLOR_GT, 2
        )

    for box in yolo_boxes:
        x, y, w, h = (np.array(box["bbox"]) * img_size[1]).astype(int)
        cv2.rectangle(
            img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 255, 255), 2
        )
        cv2.rectangle(
            img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), COLOR_YOLO, 2
        )

    for c_idx, y_idx in matches:
        c_pos = (np.array(cvat_boxes[c_idx]["bbox"][:2]) * img_size[1]).astype(int)
        y_pos = (np.array(yolo_boxes[y_idx]["bbox"][:2]) * img_size[1]).astype(int)
        cv2.line(img, tuple(c_pos), tuple(y_pos), COLOR_MATCH, 2)

    legend = [("GT", COLOR_GT), ("YOLO", COLOR_YOLO), ("Matches", COLOR_MATCH)]

    for i, (text, color) in enumerate(legend):
        cv2.putText(
            img, text, (20, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    cv2.imwrite(filename, img)


def calculate_metrics(cvat_frames, yolo_frames, iou_threshold, visualize=False):
    total_gt = total_fp = total_fn = total_iou = total_matches = 0

    for frame_idx in range(len(cvat_frames)):
        cvat_boxes = [
            {"bbox": obj["bbox"], "track_id": obj.get("track_id", -1)}
            for obj in cvat_frames[frame_idx]
        ]

        yolo_boxes_raw = [
            {
                "bbox": obj["bbox"],
                "track_id": obj.get("track_id", -1),
                "confidence": obj.get("confidence", 1.0),
            }
            for obj in yolo_frames[frame_idx]
        ]

        yolo_boxes = filter_bboxes(yolo_boxes_raw)

        matches, unmatched_cvat, unmatched_yolo = hybrid_matcher(
            cvat_boxes, yolo_boxes, iou_threshold
        )

        if visualize and frame_idx < 5:
            draw_comparison(
                cvat_boxes, yolo_boxes, matches, filename=f"match_{frame_idx}.jpg"
            )

        total_gt += len(cvat_boxes)
        total_fp += len(unmatched_yolo)
        total_fn += len(unmatched_cvat)
        total_matches += len(matches)

        for c_idx, y_idx in matches:
            total_iou += calculate_iou(
                cvat_boxes[c_idx]["bbox"], yolo_boxes[y_idx]["bbox"]
            )

    mota = 1 - (total_fp + total_fn) / max(1, total_gt)
    motp = total_iou / max(1, total_matches) if total_matches > 0 else 0
    precision = total_matches / max(1, total_matches + total_fp)
    recall = total_matches / max(1, total_gt)
    print("\nFinal Metrics:")
    print(f"GT: {total_gt}, TP: {total_matches}, FP: {total_fp}, FN: {total_fn}")
    print(f"MOTA: {mota:.4f}, MOTP: {motp:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Average IoU: {total_iou/max(1,total_matches):.4f}")
    return mota, motp


if __name__ == "__main__":
    rename_yolo_annotations("output_botsort")  # при необходимости
    cvat_xml_path = "cvat_annotations.xml"
    yolo_dir = "output_botsort"
    cvat_anns = parse_cvat_xml(cvat_xml_path)
    yolo_anns = parse_yolo_txt(yolo_dir)
    mota, motp = calculate_metrics(
        cvat_anns, yolo_anns, iou_threshold=0.3, visualize=True
    )

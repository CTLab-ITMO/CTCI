import cv2
import os
import numpy as np
import argparse

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
            print(f"Маска для {image_name} не найдена")
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

def main():
    parser = argparse.ArgumentParser(
        description='Конвертация масок в YOLO формат разметки'
    )
    
    parser.add_argument('--image_folder', required=True, 
                      help='Путь к папке с изображениями')
    parser.add_argument('--mask_folder', required=True,
                      help='Путь к папке с масками')
    parser.add_argument('--output_folder', required=True,
                      help='Путь для сохранения YOLO разметки')
    parser.add_argument('--class_id', type=int, default=0,
                      help='ID класса для разметки (по умолчанию: 0)')
    
    args = parser.parse_args()
    
    process_images_to_yolo_labels(
        image_folder=args.image_folder,
        mask_folder=args.mask_folder,
        output_folder=args.output_folder,
        class_id=args.class_id
    )

if __name__ == "__main__":
    main()
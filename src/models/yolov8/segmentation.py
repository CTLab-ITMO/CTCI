import sys

import cv2
from supervision.detection.annotate import BoxAnnotator
from supervision.detection.tools.polygon_zone import Detections
from supervision.draw.color import ColorPalette
from ultralytics import YOLO
from google.colab.patches import cv2

model = YOLO(sys.argv[2])
model.fuse()
img = cv2.imread(str(sys.argv[0]), cv2.IMREAD_ANYCOLOR)
result = model.predict(img, conf=0.85)
box_annotator = BoxAnnotator(color=ColorPalette.from_hex(['#ff0001']),
                             thickness=2, text_thickness=1, text_scale=0.4)
CLASS_NAMES_DICT = model.model.names

detections = Detections(xyxy=result[0].boxes.xyxy.cpu().numpy(), confidence=result[0].boxes.conf.cpu().numpy(),
                        class_id=result[0].boxes.cls.cpu().numpy().astype(int))
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, _, _
    in detections]

img = box_annotator.annotate(scene=img, detections=detections, labels=labels)

cv2.imwrite(sys.argv[1], img)

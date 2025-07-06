import numpy as np
from .trackingQualityAnalyzer import TrackingQualityAnalyzer
from .utils import get_center

MAX_DISAPPEARED = 3
MAX_POSITIONS_HISTORY = 5
IOU_DIST_WEIGHTS = (0.4, 0.6)
MIN_OBJECT_SIZE = 5
MIN_DISTANCE = 15
MAX_COST_THRESHOLD = 0.6
BORDER_DISAPPEAR_MULTIPLIER = 2


class ObjectTracker:
    def __init__(self, max_disappeared=1, frame_size=(1920, 1080)):
        self.next_id = 1
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.frame_width, self.frame_height = frame_size
        self.metrics = TrackingQualityAnalyzer()

    def update(self, detections, current_frame=None):
        for obj_id in self.objects:
            if len(self.objects[obj_id]["positions"]) > 1:
                last_pos = self.objects[obj_id]["positions"][-1]
                prev_pos = self.objects[obj_id]["positions"][-2]
                velocity = last_pos - prev_pos
                predicted_pos = last_pos + velocity
                self.objects[obj_id]["predicted_pos"] = predicted_pos
            else:
                self.objects[obj_id]["predicted_pos"] = get_center(
                    self.objects[obj_id]["bbox"]
                )

        for obj_id in list(self.objects.keys()):
            bbox = self.objects[obj_id]["bbox"]
            if (
                bbox[0] <= 5
                or bbox[1] <= 5
                or bbox[2] >= self.frame_width - 5
                or bbox[3] >= self.frame_height - 5
            ):
                self.objects[obj_id]["disappeared"] += BORDER_DISAPPEAR_MULTIPLIER
            else:
                self.objects[obj_id]["disappeared"] += 1

            if self.objects[obj_id]["disappeared"] > self.max_disappeared:
                del self.objects[obj_id]

        if len(detections) == 0:
            return self._prepare_output(current_frame)

        if len(self.objects) == 0:
            for det in detections:
                det_center = get_center(det)
                self.objects[self.next_id] = {
                    "bbox": det,
                    "disappeared": 0,
                    "positions": [det_center],
                    "predicted_pos": det_center,
                }
                self.next_id += 1
            return self._prepare_output(current_frame)

        obj_ids = list(self.objects.keys())
        obj_bboxes = [self.objects[obj_id]["bbox"] for obj_id in obj_ids]

        cost_matrix = np.zeros((len(obj_bboxes), len(detections)))
        for i, obj_bbox in enumerate(obj_bboxes):
            obj_predicted = self.objects[obj_ids[i]]["predicted_pos"]
            for j, det_bbox in enumerate(detections):
                det_center = get_center(det_bbox)
                iou_score = 1 - self.calculate_iou(obj_bbox, det_bbox)
                dist_score = np.linalg.norm(obj_predicted - det_center) / 100
                cost_matrix[i, j] = (
                    IOU_DIST_WEIGHTS[0] * iou_score + IOU_DIST_WEIGHTS[1] * dist_score
                )

        matched_obj_indices = set()
        matched_det_indices = set()

        while True:
            min_cost = np.min(cost_matrix)
            if min_cost > MAX_COST_THRESHOLD:
                break

            i, j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            obj_id = obj_ids[i]

            self.objects[obj_id]["bbox"] = detections[j]
            self.objects[obj_id]["disappeared"] = 0
            det_center = get_center(detections[j])
            self.objects[obj_id]["positions"].append(det_center)
            if len(self.objects[obj_id]["positions"]) > MAX_POSITIONS_HISTORY:
                self.objects[obj_id]["positions"].pop(0)

            matched_obj_indices.add(i)
            matched_det_indices.add(j)

            cost_matrix[i, :] = float("inf")
            cost_matrix[:, j] = float("inf")

        for j in set(range(len(detections))) - matched_det_indices:
            det = detections[j]
            det_center = get_center(det)

            width = det[2] - det[0]
            height = det[3] - det[1]
            if width < MIN_OBJECT_SIZE or height < MIN_OBJECT_SIZE:
                continue

            too_close = False
            for obj_id in self.objects:
                obj_center = get_center(self.objects[obj_id]["bbox"])
                distance = np.linalg.norm(obj_center - det_center)
                if distance < MIN_DISTANCE:
                    too_close = True
                    break

            if not too_close:
                self.objects[self.next_id] = {
                    "bbox": det,
                    "disappeared": 0,
                    "positions": [det_center],
                    "predicted_pos": det_center,
                }
                self.next_id += 1

        return self._prepare_output(current_frame)

    @staticmethod
    def calculate_iou(bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return inter_area / float(bbox1_area + bbox2_area - inter_area)

    def _prepare_output(self, current_frame):
        current_tracks = [
            [obj_id, obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3]]
            for obj_id, obj in self.objects.items()
        ]

        if current_frame is not None:
            self.metrics.update_metrics(
                len(self.metrics.metrics["frame"]), current_tracks, current_frame
            )
        return {obj_id: obj["bbox"] for obj_id, obj in self.objects.items()}

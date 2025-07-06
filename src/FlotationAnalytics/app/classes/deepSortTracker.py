from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
import numpy as np


class DeepSortTracker:
    def __init__(
        self,
        img_size,
        nms_max_overlap=0.6,
        max_cosine_distance=0.5,
        nn_budget=None,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
    ):
        self.img_size = img_size
        self.nms_max_overlap = nms_max_overlap
        self.iou_threshold = iou_threshold
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(metric, max_age=max_age, n_init=min_hits)

    def prepare_detections(self, yolo_detections):
        detections = []
        for det in yolo_detections:
            x1, y1, x2, y2, conf, _ = det
            bbox = (x1, y1, x2 - x1, y2 - y1)  # Конвертация в формат (x,y,w,h)
            feature = []
            detections.append(Detection(bbox, conf, feature))
        return detections

    def update(self, yolo_detections):
        detections = self.prepare_detections(yolo_detections)
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(detections)

        results = []
        for track in self.tracker.tracks:
            print("track", track, track.is_confirmed(), track.time_since_update)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return results

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
from CounTR import models_mae_cross
from PIL import Image
from torchvision import transforms
from .deepSortTracker import DeepSortTracker
from .trackingQualityAnalyzer import TrackingQualityAnalyzer
from .objectTracker import ObjectTracker

MAX_DISAPPEARED = 3
NMS_RADIUS = 12
DENSITY_THRESHOLD = 0.05
SHOT_NUM = 0
MODEL_SIZE = 384
BBOX_SIZE = 10


class VideoTracker:
    def __init__(self, model_path, tracker_type):
        self.tracker_type = tracker_type

        if tracker_type == "Мой трекер + CounTR":
            self.model = models_mae_cross.__dict__["mae_vit_base_patch16"](
                norm_pix_loss="store_true"
            )
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.custom_tracker = ObjectTracker(max_disappeared=MAX_DISAPPEARED)
            self.is_yolo_model = False
        else:
            self.model = YOLO(model_path)
            self.is_yolo_model = True

        self.quality_analyzer = TrackingQualityAnalyzer()
        self.processed_frames = []

        if tracker_type == "DeepSORT + YOLOv11s":
            self.deep_sort_tracker = DeepSortTracker(img_size=(640, 480))

    def _parse_tracks(self, results):
        tracks = []
        if self.tracker_type == "DeepSORT + YOLOv11s":
            if results[0].boxes.data is not None:
                detections = results[0].boxes.data.cpu().numpy()
                track_results = self.deep_sort_tracker.update(detections)
                tracks = [[int(t[0]), t[1], t[2], t[3], t[4]] for t in track_results]
        elif self.tracker_type == "Мой трекер + CounTR":
            if isinstance(results, dict):
                tracks = []
                for obj_id, obj_data in results.items():
                    bbox = obj_data["bbox"]
                    if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                        tracks.append([obj_id, bbox[0], bbox[1], bbox[2], bbox[3]])
                return tracks
        else:
            if hasattr(results[0], "boxes") and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                tracks = [
                    [
                        ids[i],
                        boxes[i][0],
                        boxes[i][1],
                        boxes[i][2] - boxes[i][0],
                        boxes[i][3] - boxes[i][1],
                    ]
                    for i in range(len(boxes))
                ]
        return tracks

    def _draw_detections(self, frame, tracks):
        for track in tracks:
            track_id, x, y, w, h = track
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                str(track_id),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
        return frame

    def non_max_suppression(self, points, scores, radius=NMS_RADIUS):
        sorted_indices = np.argsort(scores)[::-1]
        keep = []

        while sorted_indices.size > 0:
            i = sorted_indices[0]
            keep.append(i)

            dists = np.sqrt(
                (points[i, 0] - points[sorted_indices[1:], 0]) ** 2
                + (points[i, 1] - points[sorted_indices[1:], 1]) ** 2
            )

            to_remove = np.where(dists < radius)[0] + 1
            sorted_indices = np.delete(sorted_indices, [0] + list(to_remove))

        return points[keep]

    def process_frame(self, frame, model, old_w, old_h, tracker=None):
        if tracker is None:
            tracker = ObjectTracker(max_disappeared=MAX_DISAPPEARED)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        w, h = pil_img.size
        ratio = min(MODEL_SIZE / w, MODEL_SIZE / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

        padded_img = Image.new("RGB", (MODEL_SIZE, MODEL_SIZE), (0, 0, 0))
        pad_x = (MODEL_SIZE - new_w) // 2
        pad_y = (MODEL_SIZE - new_h) // 2
        padded_img.paste(pil_img, (pad_x, pad_y))

        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(padded_img).unsqueeze(0)
        boxes = torch.zeros(1, 0, 4)

        with torch.no_grad():
            density_map = model(img_tensor, boxes, SHOT_NUM)[0].squeeze(0).cpu().numpy()

        y, x = np.where(density_map > DENSITY_THRESHOLD * density_map.max())
        scores = density_map[y, x]
        points = np.column_stack((x, y))

        filtered_points = (
            self.non_max_suppression(points, scores) if len(points) > 0 else []
        )

        current_detections = []
        for x, y in filtered_points:
            orig_x = int((x - pad_x) * old_w / new_w)
            orig_y = int((y - pad_y) * old_h / new_h)

            x1 = max(0, orig_x - BBOX_SIZE)
            y1 = max(0, orig_y - BBOX_SIZE)
            x2 = min(old_w - 1, orig_x + BBOX_SIZE)
            y2 = min(old_h - 1, orig_y + BBOX_SIZE)
            current_detections.append((x1, y1, x2, y2))

        tracked_objects = tracker.update(current_detections, frame)

        for obj_id, bbox in tracked_objects.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                str(obj_id),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        cv2.putText(
            frame,
            f"Total: {len(tracked_objects)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        return frame, len(tracked_objects), tracker

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.processed_frames = []

        if self.tracker_type == "DeepSORT + YOLOv11s":
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.deep_sort_tracker = DeepSortTracker(
                img_size=(frame_width, frame_height)
            )

        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if self.tracker_type == "Мой трекер + CounTR":
                processed_frame, count, tracker = self.process_frame(
                    frame,
                    self.model,
                    frame.shape[1],
                    frame.shape[0],
                    self.custom_tracker,
                )
                current_tracks = self._parse_tracks(tracker.objects)
                frame_with_detections = processed_frame

                self.quality_analyzer.update_metrics(frame_num, current_tracks, frame)
            else:
                if self.tracker_type == "DeepSORT + YOLOv11s":
                    results = self.model(frame, conf=0.3, iou=0.5, max_det=400)
                    current_tracks = self._parse_tracks(results)
                else:
                    results = self.model.track(
                        frame,
                        tracker=self.tracker_type.lower().replace(
                            " + yolov11s", ".yaml"
                        ),
                        persist=True,
                        conf=0.3,
                        iou=0.5,
                        max_det=400,
                    )
                    current_tracks = self._parse_tracks(results)

                frame_with_detections = self._draw_detections(
                    frame.copy(), current_tracks
                )
                self.quality_analyzer.update_metrics(frame_num, current_tracks, frame)

            self.processed_frames.append(
                {
                    "frame": frame_with_detections,
                    "tracks": current_tracks,
                    "bubbles_count": len(current_tracks),
                }
            )

            progress_bar.progress(cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames)
            frame_num += 1

        processing_time = time.time() - start_time
        cap.release()

        metrics = self.quality_analyzer.get_final_metrics()
        metrics["final_score"] = self.quality_analyzer.get_tracking_score()

        metrics.update(
            {
                "bubbles_per_frame": [
                    f["bubbles_count"] for f in self.processed_frames
                ],
                "max_active_tracks_history": self.quality_analyzer.metrics[
                    "active_tracks"
                ],
                "track_lengths": self.quality_analyzer.metrics["track_lengths"],
                "displacement": self.quality_analyzer.metrics["displacement"],
                "coverage": self.quality_analyzer.metrics["coverage"],
                "temporal_consistency": self.quality_analyzer.metrics[
                    "temporal_consistency"
                ],
                "optical_flow": self.quality_analyzer.metrics["optical_flow"],
                "processing_time": processing_time,
            }
        )

        return metrics

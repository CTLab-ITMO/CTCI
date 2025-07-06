import cv2
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import streamlit as st


class TrackingQualityAnalyzer:
    def __init__(self):
        self.metrics = {
            "frame": [],
            "displacement": [],
            "coverage": [],
            "optical_flow": [],
            "temporal_consistency": [],
            "track_lengths": {},
            "active_tracks": [],
        }
        self.prev_tracks = {}
        self.prev_frame = None

    def update_metrics(self, frame_num, current_tracks, current_frame):
        current_dict = {}
        for track in current_tracks:
            if len(track) >= 5:
                track_id = track[0]
                bbox = track[1:5]
                current_dict[track_id] = bbox

        for track_id in current_dict:
            self.metrics["track_lengths"][track_id] = (
                self.metrics["track_lengths"].get(track_id, 0) + 1
            )

        if not self.prev_tracks:
            self.prev_tracks = current_dict
            self.prev_frame = current_frame.copy()
            return

        matched = 0
        total_displacement = 0
        total_iou = 0
        total_flow = 0

        if self.prev_frame is not None:
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

        for track_id, current_bbox in current_dict.items():
            if track_id in self.prev_tracks:
                matched += 1
                prev_bbox = self.prev_tracks[track_id]

                # Среднее смещение
                dx = current_bbox[0] - prev_bbox[0]
                dy = current_bbox[1] - prev_bbox[1]
                displacement = np.sqrt(dx**2 + dy**2)
                total_displacement += displacement

                # IoU для темпоральной согласованности
                iou = self.calculate_iou(prev_bbox, current_bbox)
                total_iou += iou

                # Анализ оптического потока в области объекта
                if self.prev_frame is not None:
                    x, y, w, h = map(int, prev_bbox)
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, current_frame.shape[1] - x), min(
                        h, current_frame.shape[0] - y
                    )
                    if w > 0 and h > 0:
                        obj_flow = flow[y : y + h, x : x + w]
                        if obj_flow.size > 0:
                            magnitude = np.sqrt(
                                obj_flow[..., 0] ** 2 + obj_flow[..., 1] ** 2
                            )
                            total_flow += np.mean(magnitude)

        # Полнота обнаружения
        coverage = matched / len(self.prev_tracks) if len(self.prev_tracks) > 0 else 0

        self.metrics["frame"].append(frame_num)
        self.metrics["displacement"].append(
            total_displacement / matched if matched > 0 else 0
        )
        self.metrics["coverage"].append(coverage)
        self.metrics["temporal_consistency"].append(
            total_iou / matched if matched > 0 else 0
        )
        self.metrics["optical_flow"].append(total_flow / matched if matched > 0 else 0)
        self.metrics["active_tracks"].append(len(current_dict))

        self.prev_tracks = current_dict
        self.prev_frame = current_frame.copy()
    
    @staticmethod
    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def get_final_metrics(self):
        if not self.metrics["frame"]:
            return {}

        avg_metrics = {
            "avg_displacement": np.mean(self.metrics["displacement"]),
            "avg_coverage": np.mean(self.metrics["coverage"]),
            "avg_temporal_consistency": np.mean(self.metrics["temporal_consistency"]),
            "avg_optical_flow": np.mean(self.metrics["optical_flow"]),
            "track_length_mean": (
                np.mean(list(self.metrics["track_lengths"].values()))
                if self.metrics["track_lengths"]
                else 0
            ),
            "track_length_median": (
                np.median(list(self.metrics["track_lengths"].values()))
                if self.metrics["track_lengths"]
                else 0
            ),
            "max_active_tracks": (
                max(self.metrics["active_tracks"])
                if self.metrics["active_tracks"]
                else 0
            ),
        }
        return avg_metrics

    def get_tracking_score(self, weights=None, normalize=True, reference_score=22.5190):
        final_metrics = self.get_final_metrics()
        print("New metrics keys:", final_metrics.keys())
        if not final_metrics:
            print("Предупреждение: нет метрик для расчета score!")
            return 0.0
        print(final_metrics)
        default_weights = {
            "avg_displacement": -0.2,
            "avg_coverage": 0.35,
            "avg_temporal_consistency": 0.25,
            "avg_optical_flow": -0.1,
            "track_length_mean": 0.2,
            "max_active_tracks": 0.1,
        }

        weights = weights if weights is not None else default_weights

        missing_metrics = [k for k in weights if k not in final_metrics]

        if missing_metrics:
            print(
                f"Предупреждение: отсутствуют метрики {missing_metrics}, они не учитываются в score"
            )

        raw_score = 0.0
        for key, weight in weights.items():
            if key in final_metrics:
                raw_score += float(final_metrics[key]) * weight

        if normalize:
            if reference_score <= 0:
                print("Ошибка: reference_score должен быть положительным!")
                return 0.0

            normalized_score = (raw_score / reference_score) - 1

            return (expit(normalized_score)) * 2
        else:
            return raw_score

    def generate_metrics_plots(self, save_path=None):
        if not self.metrics["frame"]:
            return None

        plt.figure(figsize=(16, 12))

        plt.subplot(3, 2, 1)
        plt.plot(self.metrics["frame"], self.metrics["displacement"], color="blue")
        plt.title("Динамика смещения объектов")
        plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.plot(self.metrics["frame"], self.metrics["coverage"], color="green")
        plt.title("Полнота трекинга")
        plt.grid(True)

        plt.subplot(3, 2, 3)
        plt.plot(self.metrics["frame"], self.metrics["displacement"])
        plt.plot(self.metrics["frame"], self.metrics["coverage"])
        plt.title("Качество трекинга")
        plt.grid(True)

        plt.subplot(3, 2, 4)
        plt.plot(
            self.metrics["frame"], self.metrics["temporal_consistency"], color="red"
        )
        plt.title("Темпоральная согласованность")
        plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.plot(self.metrics["frame"], self.metrics["optical_flow"], color="purple")
        plt.title("Оптический поток объектов")
        plt.grid(True)

        plt.subplot(3, 2, 6)
        plt.plot(self.metrics["frame"], self.metrics["active_tracks"], color="green")
        plt.title("Активные треки")
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return plt.gcf()

    def plot_metrics(self, save_path=None):
        fig = self.generate_metrics_plots(save_path)
        if fig is None:
            st.warning("Нет данных для построения графиков!")
            return

        st.subheader("Графики метрик")
        st.pyplot(fig)
        plt.close(fig)

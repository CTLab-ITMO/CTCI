from dataclasses import dataclass
from classes import VideoTracker
import os
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
import torch
from CounTR import models_mae_cross


@dataclass
class TrackerConfig:
    name: str
    config_file: str


@dataclass
class FlotationAnalysisConfig:
    video_path: str
    tracker: TrackerConfig
    output_dir: str = "output"
    save_frames: bool = True
    num_sample_frames: int = 5


cs = ConfigStore.instance()
cs.store(name="flotation_config", node=FlotationAnalysisConfig)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    tracker = initialize_tracker(cfg.tracker)

    print(f"Starting video processing: {cfg.video_path}")
    metrics = process_video_with_tracker(cfg.video_path, tracker)
    save_results(metrics, cfg.output_dir, cfg.save_frames, cfg.num_sample_frames)
    visualize_results(metrics, cfg.output_dir)

    print(f"Analysis completed. Results saved to: {os.path.abspath(cfg.output_dir)}")


def initialize_tracker(tracker_cfg: DictConfig):
    if tracker_cfg.name == "Мой трекер + CounTR":
        model_path = "model/FSC147.pth"
        model = models_mae_cross.__dict__["mae_vit_base_patch16"](
            norm_pix_loss="store_true"
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        tracker = VideoTracker(model_path, tracker_cfg.name)
        tracker.model = model
    else:
        tracker = VideoTracker("model/YOLOv11s.pt", tracker_cfg.name)
    return tracker


def process_video_with_tracker(video_path: str, tracker) -> dict:
    metrics = tracker.process_video(video_path)
    return metrics


def plot_metrics(metrics, output_dir):
    if "max_active_tracks_history" in metrics and metrics["max_active_tracks_history"]:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(metrics["max_active_tracks_history"], "c-", label="Активные треки")
        ax.axhline(
            y=metrics["max_active_tracks"],
            color="r",
            linestyle="--",
            label=f'Максимум: {metrics["max_active_tracks"]}',
        )
        ax.set_xlabel("Номер кадра")
        ax.set_ylabel("Количество треков")
        ax.grid(True)
        ax.legend()
        plt.savefig(os.path.join(output_dir, "active_tracks.png"))
        plt.close()

    if "optical_flow" in metrics and metrics["optical_flow"]:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(metrics["optical_flow"], "m-", label="Оптический поток")
        ax.set_xlabel("Номер кадра")
        ax.set_ylabel("Величина потока (пиксели)")
        ax.grid(True)
        ax.legend()
        plt.savefig(os.path.join(output_dir, "optical_flow.png"))
        plt.close()

    if "track_lengths" in metrics and metrics["track_lengths"]:
        fig, ax = plt.subplots(figsize=(12, 4))

        lengths = metrics["track_lengths"]
        if isinstance(lengths, dict):
            lengths = list(lengths.values())
        elif not isinstance(lengths, (list, np.ndarray)):
            lengths = []

        if lengths:
            ax.hist(lengths, bins=20, color="orange", edgecolor="black", alpha=0.7)
            mean_length = np.mean(lengths)
            ax.axvline(
                mean_length,
                color="r",
                linestyle="--",
                label=f"Среднее: {mean_length:.1f} кадров",
            )
            ax.set_xlabel("Длина трека (кадры)")
            ax.set_ylabel("Количество треков")
            ax.grid(True)
            ax.legend()
            plt.savefig(os.path.join(output_dir, "track_lengths.png"))
            plt.close()

    if "displacement" in metrics and len(metrics["displacement"]) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(metrics["displacement"], "r-", label="Смещение (пиксели)")
        ax.set_xlabel("Номер кадра")
        ax.set_ylabel("Смещение")
        ax.grid(True)
        ax.legend()
        plt.savefig(os.path.join(output_dir, "displacement.png"))
        plt.close()

    if "coverage" in metrics and len(metrics["coverage"]) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(metrics["coverage"], "g-", label="Процент совпадений")
        ax.set_xlabel("Номер кадра")
        ax.set_ylabel("Процент")
        ax.grid(True)
        ax.legend()
        plt.savefig(os.path.join(output_dir, "coverage.png"))
        plt.close()

    if "temporal_consistency" in metrics and len(metrics["temporal_consistency"]) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(metrics["temporal_consistency"], "m-", label="Средний IoU")
        ax.set_xlabel("Номер кадра")
        ax.set_ylabel("IoU")
        ax.grid(True)
        ax.legend()
        plt.savefig(os.path.join(output_dir, "temporal_consistency.png"))
        plt.close()

    if "bubbles_per_frame" in metrics and metrics["bubbles_per_frame"]:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(metrics["bubbles_per_frame"], "b-", label="Количество пузырей")
        ax.set_xlabel("Номер кадра")
        ax.set_ylabel("Количество пузырей")
        ax.grid(True)
        ax.legend()
        plt.savefig(os.path.join(output_dir, "bubbles_per_frame.png"))
        plt.close()


def save_results(
    metrics: dict, output_dir: str, save_frames: bool = True, num_sample_frames: int = 5
) -> None:
    plot_metrics(metrics, output_dir)


def visualize_results(metrics: dict, output_dir: str) -> None:
    print("\n=== Summary Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            print(f"{key}: {len(value)} elements")
        elif isinstance(value, dict):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

"""
ONNX Model Inference on Video Script

This script performs inference using an ONNX model on a video file. It reads frames from the video, applies inference
    to each frame, and displays the result in real-time.

Usage:
    python video_inference.py <config_path>

Args:
    config_path (str): Path to the YAML configuration file containing model and inference parameters.

The script initializes an ONNX session using the `init_onnx_session` function, which loads the model
    from the specified path in the configuration file.
It then reads frames from the video file, preprocesses them, performs inference using the ONNX session,
    and displays the result with the mask overlayed on the original frame.
The processing time for each frame and the average frames per second (FPS) are also printed.

"""

import time
import sys

import numpy as np
import cv2
import torchvision.transforms as transforms
import albumentations as albu

from src.models.inference import init_onnx_session
from src.models.utils.config import read_yaml_config


def draw_mask(image, mask):
    """
    Draws the mask on the input image.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Mask to be overlayed on the image.

    Returns:
        numpy.ndarray: Image with mask overlayed.
    """
    for c in (0, 1):
        mask[:, :, c] = np.zeros_like(mask[:, :, c])
    alpha_m = 1.0
    for c in range(0, 3):
        image[:, :, c] = alpha_m * mask[:, :, c] * 0.9 + image[:, :, c]

    return image


def video_onnx_inference(video_path, onnx_session):
    """
    Performs inference on a video using the provided ONNX session.

    Args:
        video_path (str): Path to the input video file.
        onnx_session (onnxruntime.InferenceSession): ONNX inference session.
    """
    vid_capture = cv2.VideoCapture(video_path)

    to_tensor = transforms.ToTensor()
    transform = albu.Compose([
        albu.CLAHE(always_apply=True),
        albu.Normalize(always_apply=True),
    ])

    if not vid_capture.isOpened():
        print("Video reading error")
        exit()
    times = []
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        (height, width) = frame.shape[:2]

        t1 = time.time()

        if not ret:
            print("Video reading error")
            exit()

        frame_to_session = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_to_session = cv2.resize(frame_to_session, (256, 256))
        frame_to_session = transform(image=frame_to_session)['image']
        frame_to_session = to_tensor(frame_to_session).unsqueeze(0).numpy()
        inputs = {'input': frame_to_session}

        outputs = onnx_session.run(None, inputs)[0][0]
        outputs = np.transpose(outputs, (1, 2, 0))
        ret, outputs = cv2.threshold(outputs, 0.5, 1, cv2.THRESH_BINARY)
        outputs = outputs * 255

        outputs = cv2.cvtColor(outputs, cv2.COLOR_GRAY2RGB)
        outputs = cv2.resize(outputs, (width, height))
        frame = draw_mask(frame, outputs)

        cv2.imshow('Bubbles', frame)

        t2 = time.time()
        times.append(t2-t1)

        key = cv2.waitKey(60)
        if (key == ord('q')) or key == 27:
            break

        if len(times) > 20:
            print("fps:", 1 / (sum(times[-20:-1]) / len(times[-20:-1])))

    vid_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    session = init_onnx_session(config_handler)
    video_onnx_inference(r".\data\test_data\bubbles\video_0.mp4", session)


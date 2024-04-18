import time
import sys

import numpy as np
import cv2
import torchvision.transforms as transforms
import albumentations as albu

from src.models.inference import init_onnx_session
from src.models.utils.config import read_yaml_config


def draw_mask(image, mask):
    for c in (0, 1):
        mask[:, :, c] = np.zeros_like(mask[:, :, c])
    alpha_m = 1.0
    for c in range(0, 3):
        image[:, :, c] = alpha_m * mask[:, :, c] * 0.9 + image[:, :, c]

    return image


def video_onnx_inference(video_path, onnx_session):
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
        times.append(t2)

        key = cv2.waitKey(60)
        if (key == ord('q')) or key == 27:
            break

    vid_capture.release()
    cv2.destroyAllWindows()
    print(sum(times)/len(times))

if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    session = init_onnx_session(config_handler)
    video_onnx_inference(r"..\data\test_data\bubbles\video_0.mp4", session)


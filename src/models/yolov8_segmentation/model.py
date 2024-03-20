import torch.nn as nn
from ultralytics import YOLO


class Yolo(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = YOLO("yolov8n-seg.pt")
    
    def train_on_batch(self, x, target):
        pass

    def forward(self, x):
        return self.net(x)

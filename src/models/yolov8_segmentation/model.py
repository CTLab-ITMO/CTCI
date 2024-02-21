from ultralytics import YOLO


def load_model(checkpoint_path: str):
    model = YOLO(checkpoint_path)
    return model


def init_model(model_yaml_path: str):
    model = YOLO(model_yaml_path)
    return model


if __name__ == "__main__":
    model = load_model("yolov8n.pt")


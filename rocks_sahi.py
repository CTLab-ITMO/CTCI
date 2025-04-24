from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="models/annotation/11rocks.pt",
    model_device="cuda:0",  # or 'cuda:0'
    model_confidence_threshold=0.4,
    source=r"D:\vscode\ctci\CTCI\data\covdor",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
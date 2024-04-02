import torch



def set_image_processor_to_datasets(model, image_size, datasets: list):
    if hasattr(model, 'image_processor'):
        image_processor = model.image_processor
        image_processor.size = image_size
        for dataset in datasets:
            dataset.image_processor = image_processor
            dataset.image_processor = image_processor


def set_gpu(device_name):
    if not torch.cuda.is_available() and device_name.split(':')[0] != 'mps' and device_name != "directml":
        device_name = 'cpu'
        print("Couldn't find gpu device. Set cpu as device")
    elif device_name == "directml":
        dml = torch_directml.device()
        return dml

    if device_name.split(':')[0] == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True

    device = torch.device(device_name)
    return device
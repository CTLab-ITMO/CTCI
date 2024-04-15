import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.models.utils.config import ConfigHandler


def export_model_onnx(
        model,
        config_handler: ConfigHandler
):
    config_export_handler = ConfigHandler(config_handler.read('export'))

    model.eval()
    input_tensor = torch.rand(*config_handler.read('input_tensor_shape'))
    torch.onnx.export(
        model,
        input_tensor,
        config_handler.read('export_path'),
        export_params=config_export_handler.read('export_params'),
        do_constant_folding=config_export_handler.read('do_constant_folding'),
        verbose=config_export_handler.read('verbose'),
        input_names=config_handler.read('input_names'),
        output_names=config_handler.read('output_names'),
        dynamic_axes=config_export_handler.read('dynamic_axes'),
        opset_version=config_export_handler.read('opset_version')
    )


def quantize_onnx(config_handler: ConfigHandler, weight_type=QuantType.QUInt8):
    model_path = config_handler.read('export_path')
    quantized_model_path = config_handler.read('acceleration', 'quantization_path')

    quantize_dynamic(
        model_path,
        quantized_model_path,
        weight_type=weight_type
    )


def init_onnx_session(config_handler):
    model_path = config_handler.read('export_path')
    providers = config_handler.read('runtime', 'providers')

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        options,
        providers
    )

    return session

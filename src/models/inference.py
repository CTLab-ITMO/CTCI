import torch
import onnxruntime as ort
import onnxruntime.quantization as quantization


def export_model_onnx(
        model,
        config_data
):
    config_data_export = config_data['export']

    model.eval()
    input_tensor = torch.rand(*config_data['input_tensor_shape'])
    torch.onnx.export(
        model,
        input_tensor,
        config_data['export_path'],
        export_params=config_data_export['export_params'],
        do_constant_folding=config_data_export['do_constant_folding'],
        verbose=config_data_export['verbose'],
        input_names=config_data['input_names'],
        output_names=config_data['output_names'],
        dynamic_axes=config_data_export['dynamic_axes'],
        opset_version=config_data_export['opset_version']
    )


def quantize_onnx(config_data, weight_type=quantization.QuantType.QInt8):
    model_path = config_data['export_path']
    quantized_model_path = config_data['acceleration']['quantization_path']

    quantization.quantize_dynamic(
        model_path,
        quantized_model_path,
        weight_type=weight_type
    )


def init_session(config_data):
    model_path = config_data['export_path']
    providers = config_data['runtime']['providers']

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        options,
        providers
    )

    return session

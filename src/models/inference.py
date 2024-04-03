import torch


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


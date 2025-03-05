import torch


def clean_state_dict(state_dict):
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "")
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def load_model(checkpoint_path):
    state_dict = torch.load(checkpoint_path, weights_only=False)
    if checkpoint_path.endswith('.ckpt'):
        state_dict = state_dict['state_dict']
    return clean_state_dict(state_dict)

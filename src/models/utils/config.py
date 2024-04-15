import yaml


class ConfigHandler:
    def __init__(self, config_data):
        self.config_data = config_data

    def read(self, *keys):
        data = self.config_data
        for key in keys:
            try:
                data = data[key]
            except KeyError:
                print(f"Cannot find key '{key}' in config file")
        return data


def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    config_handler = ConfigHandler(config_data)
    return config_handler


"""
This module provides utilities for reading and handling configuration settings
from YAML files.
"""

import yaml


class ConfigHandler:
    """
    Utility class for handling configuration data.

    Args:
        config_data (dict): Configuration data.
    """
    def __init__(self, config_data):
        self.config_data = config_data

    def check_key(self, *keys) -> bool:
        """
        Checks if the specified keys exist in the configuration data.

        Args:
            *keys: Variable number of keys.

        Returns:
            bool: True if all keys exist, False otherwise.
        """
        data = self.config_data
        is_key = True

        for key in keys:
            try:
                data = data[key]
            except KeyError:
                is_key = False

        return is_key

    def read(self, *keys):
        """
        Reads the value corresponding to the specified keys from the configuration data.

        Args:
            *keys: Variable number of keys.

        Returns:
            Any: Value corresponding to the keys.
        """
        data = self.config_data
        for key in keys:
            try:
                data = data[key]
            except KeyError:
                print(f"Cannot find key '{key}' in config file")
        return data


def read_yaml_config(file_path: str) -> ConfigHandler:
    """
    Reads a YAML configuration file and returns a ConfigHandler instance.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        ConfigHandler: Instance of ConfigHandler containing the configuration data.
    """
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    config_handler = ConfigHandler(config_data)
    return config_handler


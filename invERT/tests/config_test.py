import unittest
from invERT.config.configuration import Config
# import shutil
from pathlib import Path
from json5 import load as json_load

CONFIG_FILE = Path(f'{__file__}').parent / "config_test.json5"
with open(CONFIG_FILE, mode='r', encoding="utf8") as config_file:
    CONFIG_DICT: dict = json_load(config_file)


class TestConfig(unittest.TestCase):
    def test__get_python_type(self):
        config = Config(CONFIG_DICT)
        self.assertEqual(config._get_python_type("str"), str)
        self.assertEqual(config._get_python_type("int"), int)
        self.assertEqual(config._get_python_type("float"), float)
        self.assertEqual(config._get_python_type("bool"), bool)
        self.assertEqual(config._get_python_type("Path"), Path)
        self.assertEqual(config._get_python_type("unknown"), str)

    def test__validate_type(self):
        config = Config(CONFIG_DICT)
        value_value = CONFIG_DICT['model']['cnn']['input_channels']['value']
        value_type = CONFIG_DICT['model']['cnn']['input_channels']['type']
        key = CONFIG_DICT['model']['cnn']['input_channels']
        self.assertEqual(
            config._validate_type(
                value_value,
                value_type,
                key),
            1)
        del value_value
        del value_type
        del key

        value_value = CONFIG_DICT['model']['cnn']['biais_enabled']['value']
        value_type = CONFIG_DICT['model']['cnn']['biais_enabled']['type']
        key = CONFIG_DICT['model']['cnn']['biais_enabled']
        self.assertEqual(
            config._validate_type(
                value_value,
                value_type,
                key),
            False)
        del value_value
        del value_type
        del key

        value_value = CONFIG_DICT['model']['cnn']['conv_layers'][0][
            'filters']['value']
        value_type = CONFIG_DICT['model']['cnn']['conv_layers'][0][
            'filters']['type']
        key = CONFIG_DICT['model']['cnn']['conv_layers'][0]['filters']
        self.assertEqual(
            config._validate_type(
                value_value,
                value_type,
                key),
            8)
        del value_value
        del value_type
        del key

        value_value = CONFIG_DICT['model']['cnn']['conv_layers'][0][
            'padding']['value']
        value_type = CONFIG_DICT['model']['cnn']['conv_layers'][0][
            'padding']['type']
        key = CONFIG_DICT['model']['cnn']['conv_layers'][0]['padding']
        self.assertEqual(
            config._validate_type(
                value_value,
                value_type,
                key),
            "same")
        del value_value
        del value_type
        del key

        value_value = CONFIG_DICT['model']['cnn']['output']['value']
        value_type = CONFIG_DICT['model']['cnn']['output']['type']
        key = CONFIG_DICT['model']['cnn']['output']
        self.assertEqual(
            config._validate_type(
                value_value,
                value_type,
                key),
            Path("./output"))


if __name__ == "__main__":
    unittest.main()

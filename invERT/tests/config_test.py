import unittest
from invERT.config.configuration import Config
from pathlib import Path
from datetime import datetime
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

    def test__get_typed_value(self):
        config: Config = Config(CONFIG_DICT)
        value_value: str = CONFIG_DICT['experiment']['experiment_name'][
            'value']
        value_type: str = CONFIG_DICT['experiment']['experiment_name']['type']
        key: str = CONFIG_DICT['experiment']['experiment_name']
        self.assertEqual(
            config._get_typed_value(
                value_value,
                value_type,
                key),
            "test")
        del value_value
        del value_type
        del key

        value_value: str = CONFIG_DICT['experiment']['log']['value']
        value_type: str = CONFIG_DICT['experiment']['log']['type']
        key: str = CONFIG_DICT['experiment']['log']
        self.assertEqual(
            config._get_typed_value(
                value_value,
                value_type,
                key),
            False)
        del value_value
        del value_type
        del key

        value_value: str = CONFIG_DICT['model']['cnn']['conv_layers'][1][
            'in_channels']['value']
        value_type: str = CONFIG_DICT['model']['cnn']['conv_layers'][1][
            'in_channels']['type']
        key: str = CONFIG_DICT['model']['cnn']['conv_layers'][1]['in_channels']
        self.assertEqual(
            config._get_typed_value(
                value_value,
                value_type,
                key),
            32)
        del value_value
        del value_type
        del key

    def test_update(self):
        config = Config(CONFIG_DICT)

        overriden_dict: dict[str, str] = {
            'experiment.experiment_name': 'new_name',
            'experiment.output_folder': 'other_output',
            'experiment.log': 'true',
            'model.cnn.conv_layers[1].kernel_shape': '5',
            'model.mlp.hidden_dims': '[2, 6]',
        }

        config.update(overriden_dict)
        self.assertEqual(config.experiment.experiment_name, "new_name")
        self.assertEqual(config.experiment.output_folder, Path("other_output"))
        self.assertTrue(config.experiment.log)
        self.assertEqual(config.model.cnn.conv_layers[1].kernel_shape, 5)
        # self.assertListEqual(config.model.mlp.hidden_dims, [2, 6])

    def test_to_dict(self):
        config = Config(CONFIG_DICT)
        to_dict_config = config.to_dict(config)
        self.assertEqual(to_dict_config["experiment"]["experiment_name"],
                         "test")
        self.assertEqual(to_dict_config["experiment"]["log"], False)
        self.assertEqual(to_dict_config["model"]["cnn"]["conv_layers"][1][
            "in_channels"], 32)
        self.assertEqual(to_dict_config["model"]["mlp"]["hidden_dims"],
                         [64, 256])

    def test__get_name(self):
        config = Config(CONFIG_DICT)
        current_datetime = datetime.now().strftime("%d-%m-%Y_%Hh%M")
        save_root: Path = Path(
            CONFIG_DICT['experiment']['output_folder']['value'])
        save_folder: Path = save_root / \
            (f"{CONFIG_DICT['experiment']['experiment_name']['value']}"
             f"_{current_datetime}")
        self.assertEqual(config._get_name(), save_folder)


if __name__ == "__main__":
    unittest.main()

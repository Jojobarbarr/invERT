import unittest
from invERT.config.configuration import Config
import shutil
from pathlib import Path
import json

class TestConfig(unittest.TestCase):

    def test_initialization(self):
        config_dict = {
            "foo": "bar",
            "baz": {
                "qux": "quux",
                "quux": [
                    5,
                    3
                ]
            }
        }
        config = Config(config_dict)
        self.assertEqual(config.foo, "bar")
        self.assertEqual(config.baz.qux, "quux")
        self.assertEqual(config.baz.quux, [5, 3])

    def test_type_validation(self):
        config_dict = {
            "int_value": {"value": "10", "type": "int"},
            "float_value": {"value": "10.5", "type": "float"},
            "bool_value": {"value": "True", "type": "bool"},
            "path_value": {"value": "/some/path", "type": "Path"}
        }
        config = Config(config_dict)
        self.assertEqual(config.int_value, 10)
        self.assertEqual(config.float_value, 10.5)
        self.assertEqual(config.bool_value, True)
        self.assertEqual(config.path_value, Path("/some/path"))

    def test_update(self):
        config_dict = {
            "foo": "bar",
            "baz": {
                "qux": "quux",
                "quux": [
                    {"nested": "value"}
                ]
            }
        }
        config = Config(config_dict)
        updates = {
            "foo": "new_bar",
            "baz.qux": "new_quux",
            "baz.quux[0].nested": "new_value"
        }
        config.update(updates)
        self.assertEqual(config.foo, "new_bar")
        self.assertEqual(config.baz.qux, "new_quux")
        self.assertEqual(config.baz.quux[0].nested, "new_value")

    def test_save(self):
        expected_path = Path("/tmp/test_output")
        config_dict = {
            "experiment": {
                "output_folder": expected_path
            }
        }
        config = Config(config_dict)
        try:
            self.assertTrue(config.save())
            with open(expected_path / "config.json", 'r', encoding="utf8") as f:
                saved_config = json.load(f)
            saved_path = Path(saved_config["experiment"]["output_folder"])
            self.assertEqual(saved_path, expected_path)
        finally:
            shutil.rmtree(expected_path, ignore_errors=True)



if __name__ == "__main__":
    unittest.main()

from json import dump as json_dump
from pathlib import Path

class Config:
    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            if isinstance(value, dict) and "value" in value and "type" in value:
                value = self._validate_type(value["value"], value["type"], key)
            elif isinstance(value, dict):
                # Recursively turn dictionaries into Config objects
                value = Config(value)
            elif isinstance(value, list):
                # Recursively turn lists of dictionaries into lists of Config objects
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        value[index] = Config(item)
            setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)
    
    def _validate_type(self, value: str, expected_type: str, key: str):
        """Validate the type of a configuration value."""
        python_type = self._get_python_type(expected_type)
        try:
            value = python_type(value)
        except ValueError:
            raise ValueError(f"Key '{key}' expected {expected_type}, got {value}")
        return value
    
    def _get_python_type(self, type_str: str) -> type:
        """Convert a string type to an actual Python type. If the type is not recognized, return str."""
        types_map: dict[str: type] = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "Path": Path
        }
        return types_map.get(type_str, str)

    def update(self, updates: dict):
        for key, value in updates.items():
            keys: list[str] = key.split('.')
            sub_config: Config = self
            for sub_key in keys[:-1]:
                if '[' in sub_key:
                    sub_key, index = sub_key.strip(']').split('[')
                    sub_config = getattr(sub_config, sub_key)[int(index)]
                else:
                    sub_config = getattr(sub_config, sub_key)
            setattr(sub_config, keys[-1], value)
    
    def save(self):
        def to_dict(obj):
            """Recursively convert Config objects to dictionaries."""
            if isinstance(obj, Config):
                return {key: to_dict(value) for key, value in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [to_dict(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: to_dict(value) for key, value in obj.items()}
            return obj
        
        try:
            self.experiment.output_folder.mkdir(parents=True)
        except FileExistsError:
            keep_going: str = input(f"Output folder already exists: {self.experiment.output_folder}, do you want to continue? (y/n) ")
            if keep_going.lower() != 'y':
                return False
        with open(self.experiment.output_folder / "config.json", 'w', encoding="utf8") as f:
            json_dump(to_dict(self), f, indent=2)
        return True

if __name__ == "__main__":
    config = Config({"foo": "bar",
        "baz": {
            "qux": "quux",
            "quux": [
                5,
                3
            ]
        }
    })
    print(config.foo)  # bar
    print(config.baz.qux)  # quux
    print(config.baz.quux)  # [5, 3]
    print(config)  # {'foo': 'bar', 'baz': {'qux': 'quux'}}
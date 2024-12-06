from json5 import dump as json_dump
from pathlib import Path
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime


class Config:
    def __init__(self, config_dict_arg: dict):
        config_dict = deepcopy(config_dict_arg)
        for key, value in config_dict.items():
            if isinstance(
                    value,
                    dict) and "value" in value and "type" in value:
                value = self._validate_type(value["value"], value["type"], key)
            elif isinstance(value, dict):
                # Recursively turn dictionaries into Config objects
                value = Config(value)
            elif isinstance(value, list):
                # Recursively turn lists of dictionaries into lists of Config
                # objects
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
        except ValueError as e:
            raise ValueError(
                f"\nKey '{key}' expected {expected_type}, got value {value} of type {type(value)} -- Original error: {e}\n")
        return value

    def _get_python_type(self, type_str: str) -> type:
        """Convert a string type to an actual Python type. If the type is not recognized, return str."""
        types_map: dict[str: type] = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
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
            python_type: type = type(getattr(sub_config, keys[-1]))
            setattr(sub_config, keys[-1], python_type(value))

    def save(self, args: ArgumentParser) -> bool:
        def to_dict(obj: dict) -> dict:
            """Recursively convert Config objects to dictionaries."""
            if isinstance(obj, Config):
                return {key: to_dict(value)
                        for key, value in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [to_dict(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: to_dict(value) for key, value in obj.items()}
            return obj

        if not self.check():
            return False
        # Save foler is experiment_name_date_time
        # Get the current date and time
        current_datetime = datetime.now()

        # Format the date and time
        formatted_datetime: str = current_datetime.strftime("%d-%m-%Y_%Hh%M")
        save_folder: Path = self.experiment.output_folder / \
            f"{self.experiment.experiment_name}_{formatted_datetime}"
        self.experiment.output_folder = save_folder
        try:
            self.experiment.output_folder.mkdir(parents=True)
        except FileExistsError:  # There is already an experiment with the same name, warns the user,
            # and if flag --yes is not True, asks for confirmation before
            # overwriting it.
            if not args.yes:
                keep_going: str = input(
                    f"Output folder already exists here {self.experiment.output_folder.resolve()}. "
                    f"Do you want to continue? (y/n) ")
                if keep_going.lower() != 'y':
                    return False
        with open(self.experiment.output_folder / "config.json5", 'w', encoding="utf8") as f:
            json_dump(to_dict(self), f, indent=2)
        return True

    def check(self) -> bool:
        """Check if the configuration is valid."""
        limit: float = 0.9
        assert self.dataset.test_split + \
            self.dataset.validation_split < limit, \
            (f"The sum of test_split and validation_split must be less than {limit}. "
             f"You have a test_split = {self.dataset.test_split} and a validation_split = {self.dataset.validation_split} "
             f"(sum is {self.dataset.test_split + self.dataset.validation_split}.")

        output_filter: int = 1
        assert self.model.cnn.conv_layers[
            -1].filters == output_filter, \
            (f"The number of filters in the last convolutional layer must be {output_filter}. "
             f"You have {self.model.cnn.conv_layers[-1].filters}.")

        implemented_optimizers: list[str] = ["adam", "sgd", "rmsprop"]
        assert self.training.optimizer in implemented_optimizers, \
            f"Optimizer must be one of {implemented_optimizers}. You have {self.training.optimizer}."

        implemented_training_losses: list[str] = ["mse", "l1"]
        assert self.training.loss_function in implemented_training_losses, \
            f"Loss must be one of {implemented_training_losses}. You have {self.training.loss_function}"

        implemented_lr_schedulers: list[str] = ["plateau"]
        assert self.training.lr_scheduler.type in implemented_lr_schedulers, \
            (f"Learning rate scheduler type must be one of {implemented_lr_schedulers}. "
             f"You have {self.training.lr_scheduler.type}.")

        iteration_per_epoch: int = int(
            (self.dataset.num_samples *
             (
                 1 -
                 self.dataset.test_split -
                 self.dataset.validation_split)) //
            self.dataset.batch_size)
        assert self.logging.print_points < iteration_per_epoch, \
            (f"print_points must be less than the number of iterations in an epoch. "
             f"You have {self.logging.print_points} for {iteration_per_epoch} iterations per epoch.")
        return True

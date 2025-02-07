from json5 import dump as json_dump
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import TypeVar

T = TypeVar('T', str, int, float, bool, Path)


class Config:
    def __init__(self,
                 config_dict_arg: dict
                 ):
        """
        Create a Config object from a json5 loaded dictionary.

        The Config object is created by recursively turning dictionaries into
        Config objects and lists of dictionaries into lists of Config objects.
        Until the value is a dictionary with a "value" and a "type" key, the
        value is validated to be of the expected type. Ultimately, the
        attributes of the Config object are nested Config objects or lists of
        Config objects.

        @param config_dict_arg: The dictionary loaded from the json5 file.
        """
        config_dict: dict = deepcopy(config_dict_arg)
        for key, value in config_dict.items():
            if isinstance(value, dict):
                if "value" in value and "type" in value:
                    # Smallest granularity here, the value["value"]
                    # is the value of this attribute
                    value = self._get_typed_value(value["value"],
                                                  value["type"],
                                                  key)
                else:
                    # Recursively turn dictionaries into Config objects
                    value = Config(value)
            elif isinstance(value, list):
                # Recursively turn lists of dictionaries into lists of Config
                # objects
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        value[index] = Config(item)
            else:
                raise ValueError(
                    f"Expected a dictionary or a list, got {value} of type "
                    f"{type(value)} for key {key}.")
            # Set the attribute of the Config object
            setattr(self, key, value)

    def _get_typed_value(self,
                         value: str,
                         expected_type: str,
                         key: str,
                         ) -> T:
        """Get a typed value from a string value.

        If the value cannot be converted to the expected type, raise a
        ValueError with a message indicating the key, the expected type, the
        value and the original error message.

        @param value: The value to convert.
        @param expected_type: The expected type of the value. Should be one of
        "str", "int", "float", "bool", "Path".
        @param key: The key of the value in the configuration file.
        @return: The value converted to the expected type T (str, int, float,
        bool, Path).
        """
        python_type: type = self._get_python_type(expected_type)
        if python_type == bool:
            if value.lower() == "true":
                value = True
            else:
                value = False
        else:
            try:
                value: T = python_type(value)
            except ValueError as e:
                raise ValueError(
                    f"\nKey '{key}' expected to be of type{expected_type}, "
                    f"but got value {value} of type {type(value)} -- "
                    f"Original error: \n{e}\n")
        return value

    def _get_python_type(self,
                         type_str: str
                         ) -> type:
        """
        Convert a string type to an actual Python type.

        If the type is not recognized, return str.

        @param type_str: The string representation of the type.
        @return: The Python type.
        """
        types_map: dict[str, type] = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "Path": Path
        }
        return types_map.get(type_str, str)  # Default to str if not found

    def update(self,
               updates: dict[str, str]
               ) -> None:
        """
        Update the Config object with new values.

        The updates dictionary should have the format
        "'key1.key2.key3': value". The function will recursively navigate the
        Config object until the second to last key, and then update the last
        key with the new value.

        @param updates: The dictionary of updates. The keys are strings with
        the format "key1.key2.key3", and the values are the new values of
        type str.

        @return: None

        TODO: Add support for updating lists and dictionaries in the Config
        """
        for key, value in updates.items():
            # key is a string with the format "key1.key2.key3"
            keys: list[str] = key.split('.')
            # Start from the top level Config object, self.
            sub_config: Config = self
            for sub_key in keys[:-1]:
                if '[' in sub_key:
                    # Here, it means we are dealing with a list of the form:
                    # 'key[index]'. We need to split the key by '[', get rid
                    # of the last ']', and get the index.
                    sub_key, index = sub_key.strip(']').split('[')
                    # Now we just need to get the attribute at the index and
                    # keep going.
                    sub_config = getattr(sub_config, sub_key)[int(index)]
                else:
                    sub_config = getattr(sub_config, sub_key)
            # Get the type of the value to update: this is the type of the
            # sub_config object attribute associated to the second to last
            # key.
            python_type: type = type(getattr(sub_config, keys[-1]))
            # Update the last key with the typed new value.
            setattr(sub_config, keys[-1], python_type(value))

    def to_dict(self,
                obj: "Config" | T
                ) -> dict[dict, dict | list | T]:
        """
        Recursively convert Config objects to dictionaries.

        The function will convert Config objects to dictionaries and lists of
        Config objects to lists of dictionaries. Other types are values of the
        attributes of the Config object, so they are left as they are.
        The function goes recursively through the object to be sure to convert
        every sub-Config object to dictionaries.

        @param obj: The object to convert to a dictionary. It can be a Config
        object, or any other types authorized in a Config object.
        """
        if isinstance(obj, Config):
            # Recursively convert Config objects to dictionaries
            return {key: self.to_dict(value)
                    for key, value in obj.__dict__.items()}
        elif isinstance(obj, list):
            # Recursively convert lists of Config objects to lists of
            # dictionaries
            return [self.to_dict(item) for item in obj]
        elif isinstance(obj, Path):
            # Convert Path objects to strings
            return str(obj)
        elif isinstance(obj, dict):
            # Goes through the dictionary to be sure to convert an eventual
            # Config object inside the dictionary.
            return {key: self.to_dict(value) for key, value in obj.items()}
        # Other types are values of the attributes of the Config object, so
        # they are left as they are. Or they are the final value of the
        # dictionary.
        return obj

    def _get_name(self,
                  ) -> str:
        """
        Get the name of the save folder ('experiment_name_date_time').

        The name of the experiment is the concatenation of the experiment
        name and the datetime of execution.

        @return: The name of the experiment.
        """
        # Get the current date and time
        current_datetime = datetime.now().strftime("%d-%m-%Y_%Hh%M")

        # The path of the save folder is the output folder of the experiment
        # with the name of the experiment and the date and time of execution
        save_folder: Path = self.experiment.output_folder / \
            f"{self.experiment.experiment_name}_{current_datetime}"
        return save_folder

    def _create_save_folder(self,
                            always_yes: bool,
                            ) -> bool:
        """
        Create the save folder for the experiment.

        The save folder is created with the name of the experiment and the
        date and time of execution. If the folder already exists, the function
        asks for confirmation before overwriting it. If the flag --yes is
        used, the function does not ask for confirmation. If the folder is
        successfully created, the function returns True. Otherwise, it returns
        False.

        @param always_yes: Skip the confirmation prompt if True.
        @return: True if the save folder is successfully created, False
        otherwise.
        """
        try:
            self.experiment.output_folder.mkdir(parents=True)
        except FileExistsError:
            # There is already an experiment with the same name.
            # Warns the user, and if flag --yes is not True, asks
            # for confirmation before overwriting it.
            if not always_yes:
                keep_going: str = input(
                    f"Output folder already exists here "
                    f"{self.experiment.output_folder.resolve()}. "
                    f"Do you want to continue and potentially overwite it? "
                    f"(y/n) ")
                if keep_going.lower() != 'y':
                    return False
        return True

    def check_and_save(self,
                       always_yes: bool,
                       ) -> bool:
        """
        Check the validity of the configuration and save it to JSON5 format.

        The configuration is checked for validity using the check method. If
        the configuration is valid, the save method is called to save the
        configuration to a JSON5 file in the output folder specified in the
        configuration file. If the configuration is not valid, the function
        returns False.

        @param always_yes: Skip the confirmation prompt if True.
        @return: True if the configuration file is saved successfully, False
        otherwise.

        @raise AssertionError: If the configuration is not valid.
        """
        if not self._check():  # Check if the configuration is valid
            return False

        # Overwrite save foler to be 'experiment_name_date_time'
        self.experiment.output_folder = self._get_name()

        # Create the save folder
        if not self._create_save_folder(always_yes):
            return False

        # Save the configuration file
        with open(self.experiment.output_folder / "config.json5", 'w',
                  encoding="utf8") as config_file:
            json_dump(self.to_dict(self), config_file, indent=2)
        return True

    def _check(self
               ) -> bool:
        """
        Check if the configuration is valid.

        If the configuration is not valid, the function raises an
        AssertionError with a message indicating the issue.
        """
        # Check if the sum of test_split and validation_split is less than 0.4
        limit: float = 0.4
        data_left: int = self.dataset.test_split - \
            self.dataset.validation_split
        assert (data_left < limit), \
            (f"The sum of test_split and validation_split must be less than "
             f"{limit}. You have a test_split = {self.dataset.test_split} "
             f"and a validation_split = {self.dataset.validation_split} "
             f"(sum is {data_left}).")

        # Check if the selected optimizer is implemented
        implemented_optimizers: list[str] = ["adam", "sgd", "rmsprop"]
        assert self.training.optimizer.type in implemented_optimizers, \
            (f"Optimizer must be one of {implemented_optimizers}. "
             f"You have {self.training.optimizer}.")

        # Check if the selected loss function is implemented
        implemented_training_losses: list[str] = ["mse", "l1"]
        assert self.training.loss_function in implemented_training_losses, \
            (f"Loss must be one of {implemented_training_losses}. You have "
             f"{self.training.loss_function}")

        # Check if the selected learning rate scheduler is implemented
        implemented_lr_schedulers: list[str] = ["plateau"]
        assert self.training.lr_scheduler.type in implemented_lr_schedulers, \
            (f"Learning rate scheduler type must be one of "
             f"{implemented_lr_schedulers}. You have "
             f"{self.training.lr_scheduler.type}.")

        # Check if the batch_mixture is a divisor of the batch_size
        batch_size = self.dataset.batch_size
        batch_mixture = self.dataset.batch_mixture
        assert batch_size % batch_mixture == 0, \
            (f"batch_mixture must be a divisor of batch_mixture. "
             f"You have {batch_size} and {batch_mixture}.")

        # Check if num_sub_group is a divisor of num_samples
        num_samples = self.dataset.num_samples
        num_sub_group = self.dataset.num_sub_group
        assert num_samples % num_sub_group == 0, \
            (f"num_sub_group must be a divisor of num_samples. "
             f"You have {num_samples} and {num_sub_group}.")

        # Check if the test sub-groups size are greater than 0
        test_sub_group_size = self.dataset.test_split \
            * (num_samples // num_sub_group)
        assert test_sub_group_size > 0, \
            (f"test sub_groups size must be greater than 0. "
             f"Your test sub_groups size is {test_sub_group_size}.\n"
             f"Increase test_split, dataset_size or decrease num_sub_group.")

        # Check if the validation sub-groups size are greater than 0
        val_sub_group_size = self.dataset.validation_split \
            * (num_samples // num_sub_group)
        assert val_sub_group_size > 0, \
            (f"validation sub_groups size must be greater than 0. "
             f"Your validation sub_groups size is {val_sub_group_size}.\n"
             f"Increase validation_split, dataset_size or decrease "
             f"num_sub_group.")

        # Check if the batch_mixture is less than or equal to the number of
        # sub-groups
        assert batch_mixture <= num_sub_group, \
            (f"batch_mixture must be less than or equal to num_sub_group. "
             f"You have {batch_mixture} and {num_sub_group}.")

        # Check if logging.print_points is a divisor of the number of batches.
        iteration_per_epoch = num_samples // batch_size
        assert iteration_per_epoch % self.logging.print_points == 0, \
            (f"logging.print_points must be a divisor of the number of "
             f"batches. You have {iteration_per_epoch} and "
             f"{self.logging.print_points}.")

        return True

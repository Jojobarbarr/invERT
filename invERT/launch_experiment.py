from runner import main as run_experiment
from config.configuration import Config
from pathlib import Path
from json5 import load as json_load
from argparse import ArgumentParser
from typing import Any


def parse_arguments() -> ArgumentParser:
    """
    Parse command line arguments.

    The arguments are:
    - config_file: Path to the configuration file (JSON format). MANDATORY.
    - debug: Enable debug mode. Works with logging module. OPTIONAL.
    - yes: Skip confirmation prompt. OPTIONAL.
    - override: Override specific config parameters
    (e.g., training.epochs=100). TODO: May need some verification about some
    data types (lists, layer?). OPTIONAL.

    Returns:
    --------
    ArgumentParser
        The parsed command line arguments.
    """
    parser = ArgumentParser(
        description=(
            "Launch an experiment with a specified configuration file."
        )
    )
    parser.add_argument(
        'config_file',
        type=Path,
        help="Path to the configuration file (JSON format).")
    parser.add_argument(
        '-o',
        '--override',
        nargs='+',
        help=(
            "Override specific config parameters "
            "(e.g., training.epochs=100)."
        ),
        default=[])
    return parser.parse_args()


def load_config(config_file: Path
                ) -> Config | None:
    """
    Load the configuration file and return the Config instance.

    The configuration file should be a JSON5 file. If the file is not found,
    the function will print an error message and return None.

    Parameters
    ----------
    config_file : Path
        The path to the configuration file.
    
    Returns
    -------
    Config | None
        The Config instance if the file is loaded successfully, None otherwise.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    """
    try:
        with open(config_file, mode='r', encoding="utf8") as config_file:
            config_dict: dict[str, Any] = json_load(config_file)
            config: Config = Config(config_dict)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}, exiting.")
        return None
    print(f"\nSuccessfully loaded configuration file: {config_file.name}\n")
    return config


def override_config(overriden_arguments: list[str],
                    ) -> dict[str, str]:
    overriden_dict: dict[str, str] = {}
    for overriden_idx, overriden_argument in enumerate(overriden_arguments):
        overriden_argument_split: list[str] = overriden_argument.split('=')

        assert len(overriden_argument_split) == 2, \
            (f"If an override argument is provided, it should be in the form "
             f"of 'key=value'. Here, you provided: '{overriden_argument}' at "
             f"index {overriden_idx} of the override list.")

        overriden_dict[overriden_argument_split[0]] = \
            overriden_argument_split[1]
    return overriden_dict


def save_config(config: Config,
                ) -> bool:
    """
    Call the save method of the Config instance to save the configuration
    file and returns False if saving has an issue.

    The configuration file is saved as a JSON5 file in the output folder
    specified in the configuration file. Refer to the Config class for more
    information.

    Parameters
    ----------
    config : Config
        The Config instance containing the configuration to be saved.
    
    Returns
    -------
    bool
        True if the configuration file is saved successfully, False otherwise.
    """
    if not config.check_and_save():
        print(
            f"\nFailed to save configuration file to "
            f"{config.experiment.output_folder.resolve()}, exiting.")
        return False
    print(
        f"\nSuccessfully saved updated configuration file to "
        f"{config.experiment.output_folder / 'config.json5'}\n")
    return True


def main():
    args: ArgumentParser = parse_arguments()

    # Load the configuration file
    config: Config | None = load_config(args.config_file)
    if config is None:
        return

    # Overrides if any override arguments are provided
    if len(args.override) > 0:
        overrides: dict[str, str] = override_config(args.override)
        config.update(overrides)

    # Save the updated config to the experiment result folder
    save_config(config)

    # Run the experiment
    run_experiment(config)


if __name__ == "__main__":
    main()

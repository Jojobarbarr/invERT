from runner import main as run_experiment
from config.configuration import Config
from pathlib import Path
from json5 import load as json_load
from argparse import ArgumentParser
import logging
import time
print("Importing...")
start_time = time.perf_counter()
print(f"Importation done in {time.perf_counter() - start_time:.2f} seconds")


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

    @return: The ArgumentParser instance.
    """
    parser = ArgumentParser(
        description=("Launch an experiment with a specified configuration "
                     "file."))
    parser.add_argument(
        'config_file',
        type=Path,
        help="Path to the configuration file (JSON format).")
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help="Enable debug mode.")
    parser.add_argument(
        '-y',
        '--yes',
        action='store_true',
        help="Skip confirmation prompt.")
    parser.add_argument(
        '-o',
        '--override',
        nargs='+',
        help=("Override specific config parameters "
              "(e.g., training.epochs=100)."),
        default=[])
    return parser.parse_args()


def init_logging(debug: bool
                 ) -> None:
    """
    Initialize the logging configuration.

    Makes sure that the logging configuration is reset before setting it up if
    debug mode is enabled. Otherwise, the INFO level is set.

    @param debug: Enable debug mode if True.
    """
    # Reset logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="\n%(asctime)s - %(levelname)s -\n%(message)s\n"
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s"
        )

    logging.debug("Debug messages are printed.")
    logging.info("Info messages are printed.")

    return


def load_config(config_file: Path
                ) -> Config | None:
    """
    Load the configuration file and return the Config instance.

    The configuration file should be a JSON5 file. If the file is not found,
    the function will print an error message and return None.

    @raise FileNotFoundError: If the configuration file is not found.

    @param config_file: Path to the configuration file.
    @return: The Config instance.
    """
    try:
        with open(config_file, mode='r', encoding="utf8") as config_file:
            config_dict: dict = json_load(config_file)
            config: Config = Config(config_dict)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}, exiting.")
        return None
    print(f"\nSuccessfully loaded configuration file: {config_file.name}\n")
    return config


def override_config(overriden_arguments: list[str],
                    ) -> dict[str, str]:
    for overriden_idx, overriden_argument in enumerate(overriden_arguments):
        overriden_argument_split: list[str] = overriden_argument.split('=')

        assert len(overriden_argument_split) == 2, \
            (f"If an override argument is provided, it should be in the form "
             f"of 'key=value'. Here, you provided: '{overriden_argument}' at "
             f"index {overriden_idx} of the override list.")

        overriden_dict: dict[str, str] = {
            overriden_argument_split[0]: overriden_argument_split[1]
        }
    return overriden_dict


def save_config(config: Config,
                always_yes: bool,
                ) -> bool:
    """
    Call the save method of the Config instance to save the configuration
    file and returns False if saving has an issue.

    The configuration file is saved as a JSON5 file in the output folder
    specified in the configuration file. Refer to the Config class for more
    information.

    @param config: The Config instance.
    @param always_yes: Skip the confirmation prompt if True.
    @return: True if the configuration file is saved successfully, False
    otherwise.
    """
    if not config.check_and_save(always_yes):
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

    # Initialize logging
    init_logging(args.debug)

    # Load the configuration file
    config: Config | None = load_config(args.config_file)
    if config is None:
        return

    # Overrides if any override arguments are provided
    if len(args.override) > 0:
        overrides: dict[str, str] = override_config(args.override)
        config.update(overrides)

    # Save the updated config to the experiment result folder
    save_config(config, args.yes)

    # Run the experiment
    run_experiment(config)


if __name__ == "__main__":
    main()

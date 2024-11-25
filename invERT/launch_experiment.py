from argparse import ArgumentParser
from json5 import load as json_load
from pathlib import Path
from config.configuration import Config


def parse_arguments():
    parser = ArgumentParser(description="Launch an experiment with a specified configuration file.")
    parser.add_argument('config_file', type=Path, help="Path to the configuration file (JSON format).")
    parser.add_argument('--override', nargs='+', help="Override specific config parameters (e.g., training.epochs=100).", default=[])
    return parser.parse_args()

def main():
    args: ArgumentParser = parse_arguments()

    # Load the configuration file
    try:
        with open(args.config_file, mode='r', encoding="utf8") as config_file:
            config_dict: dict = json_load(config_file)
            config: Config = Config(config_dict)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config_file}, exiting.")
        return
    print(f"\nSuccessfully loaded configuration file: {args.config_file}\n")

    # Process overrides
    overrides: dict[str: str] = {overriden_arguments.split('=')[0]: overriden_arguments.split('=')[1] for overriden_arguments in args.override}
    config.update(overrides)

    # Save the updated config to the experiment result folder
    if not config.save():
        print(f"\nFailed to save configuration file to {config.experiment.output_folder.resolve()}, exiting.")
        return
    print(f"\nSuccessfully saved updated configuration file to {config.experiment.output_folder / 'config.json5'}\n")

    for repetition in range(config.experiment.repetitions):
        print(f"Starting repetition {repetition + 1} of experiment: {config.experiment.experiment_name}")
        # Run the experiment
        # run_experiment(config)


if __name__ == "__main__":
    main()

import time
print("Importing...")
start_time = time.perf_counter()
import logging
from argparse import ArgumentParser
from json5 import load as json_load
from pathlib import Path
from config.configuration import Config
from runner import main as run_experiment
print(f"Importation done in {time.perf_counter() - start_time:.2f} seconds")



def parse_arguments():
    parser = ArgumentParser(description="Launch an experiment with a specified configuration file.")
    parser.add_argument('config_file', type=Path, help="Path to the configuration file (JSON format).")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode.")
    parser.add_argument('-o', '--override', nargs='+', help="Override specific config parameters (e.g., training.epochs=100).", default=[])
    return parser.parse_args()

def main():
    args: ArgumentParser = parse_arguments()

    # Reset logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.debug:
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

    run_experiment(config)



if __name__ == "__main__":
    main()

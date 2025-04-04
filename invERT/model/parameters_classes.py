from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoggingParameters:
    loss_value: list[float]
    test_loss_value: list[float]
    print_points: set[int]
    print_points_list: list[int]
    batch_size: int
    figure_folder: Path
    model_output_folder: Path

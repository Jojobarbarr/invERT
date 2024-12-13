from dataclasses import dataclass
import multiprocessing as mp
import numpy as np


@dataclass
class TestingParameters:
    loss_arrays: np.ndarray
    test_loss_arrays: np.ndarray
    repetition: int
    print_points: int
    nb_print_points: int
    queue: mp.Queue
    epoch: int
    batch_loss: float

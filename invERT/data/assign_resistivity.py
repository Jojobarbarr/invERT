import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import perf_counter
import math
import pygimli as pg
import pygimli.physics.ert as ert


def detransform(log_res: float | np.ndarray[float]) -> float | np.ndarray[float]:
    return 2 * 10 ** (4 * log_res)


def compute_active_columns(row: int, is_even_row: bool, total_cols: int, offset: int) -> tuple[int, int]:
    """
    Compute the starting and ending column indices for the given row.
    """
    if is_even_row:
        col_start = math.ceil(row * 1.5) - offset
        col_end = total_cols - math.ceil(row * 1.5) + offset
    else:
        col_start = math.ceil((row + 1) * 1.5) - 1 - offset
        col_end = total_cols - math.ceil((row + 1) * 1.5) + 1 + offset
    return col_start, col_end


if __name__ == "__main__":
    dataset_path: Path = Path("../../../dataset/clean_unified")
    output_path: Path = Path("../../../dataset/processed")

    random_gen: np.random.Generator = np.random.default_rng()

    nbr_npz: int = len(list(dataset_path.glob("*.npz")))
    random_npz: int = random_gen.integers(0, nbr_npz - 1, 1)[0]

    multi_array: np.ndarray[np.int8] = np.load(dataset_path / f"{random_npz}.npz")["arr_0"]
    section_id: int = random_gen.choice(multi_array.shape[0], 1)[0]
    section: np.ndarray[np.int8] = multi_array[section_id]

    start = perf_counter()

    nbr_electrodes: int = random_gen.integers(24, 96, 1)[0]

    pixel_length: np.float64 = random_gen.integers(1, 20, 1)[0] / 2

    total_true_length: np.float64 = nbr_electrodes * pixel_length
    total_pixels_to_keep: int = (nbr_electrodes - 1) * 2

    sample: np.ndarray[np.int8] = section[:total_pixels_to_keep // 2, :total_pixels_to_keep]

    rock_classes: np.ndarray[np.int8] = np.unique(sample)
    sample_log_res: np.ndarray[np.float64] = sample.astype(np.float64)
    for rock_class in rock_classes:
        random_log_resistivity: np.float64 = random_gen.uniform(0, 1)
        sample_log_res[sample == rock_class] = random_log_resistivity
    
    x_arr: np.ndarray[np.float64] = np.linspace(0., total_pixels_to_keep, total_pixels_to_keep + 1, dtype=np.float64)
    y_arr: np.ndarray[np.float64] = np.linspace(-(total_pixels_to_keep // 2), 0, total_pixels_to_keep // 2 + 1, dtype=np.float64)
    world: pg.core.Mesh = pg.createGrid(x=x_arr, y=y_arr, worldBoundaryMarker=True)

    scheme_names: list[str] = ["wa", "slm"]

    elec_array: np.float64 = np.linspace(0., total_pixels_to_keep, nbr_electrodes, dtype=np.float64)  # np.float64 to be compatible with C++ double
    schemes: dict[str, pg.DataContainerERT] = {scheme_name: ert.createData(elecs=elec_array, schemeName=scheme_name) for scheme_name in scheme_names}

    fig, axes = plt.subplots(len(scheme_names), 1, figsize=(10, 10))
    for idx, (key, scheme) in enumerate(schemes.items()):
        pg.show(world, ax=axes[idx], showMesh=True)
        electrode_positions: np.ndarray[np.float64] = np.array([scheme.sensorPosition(i) for i in range(scheme.sensorCount())])
        axes[idx].scatter(electrode_positions[:, 0], electrode_positions[:, 1], marker='+', color='red', label='Electrodes', zorder=3)
        axes[idx].set_title(f"Electrode positions for {key} scheme")

    sample_res: np.ndarray[np.float64] = detransform(sample_log_res)
    results: dict[str, pg.DataContainerERT] = {electrode_scheme_name: ert.simulate(world, res=sample_res.ravel(), scheme=scheme, verbose=True) for electrode_scheme_name, scheme in schemes.items()}

    # Extract the apparent resistivity values for the Wenner array
    result_wenner_array: pg.DataContainerERT = results["wa"]

    rhoa: list[float] = result_wenner_array["rhoa"]
    num_values: int = len(rhoa)

    num_rows: int = (nbr_electrodes - 1) // 3
    num_cols: int = nbr_electrodes - 3

    even_num_cols: bool = (num_cols % 2 == 0)
    if even_num_cols:
        num_cols += 1

    offset: int = (nbr_electrodes - 1) % 2

    result: np.ndarray[np.float64] = np.zeros((num_rows, num_cols), dtype=np.float64)
    value_index: int = 0

    for i in range(num_rows):
        # Determine if the current row is considered "even" based on num_cols parity
        is_even_row = (i % 2 == 0) if even_num_cols else (i % 2 == 1)
        
        col_start, col_end = compute_active_columns(i, is_even_row, num_cols, offset)
        
        for j in range(col_start, col_end):
            # For even rows, use a special average at the center column
            if is_even_row and j == (num_cols - 1) // 2:
                result[i, j] = (rhoa[value_index - 1] + rhoa[value_index]) / 2
            else:
                result[i, j] = rhoa[value_index]
                value_index += 1
    
    # Extract the apparent resistivity values for the Schlumberger array
    result_schlumberger_array: pg.DataContainerERT = results["slm"]

    # Get the data and print the number of elements
    rhoa: list[float] = result_schlumberger_array["rhoa"]
    num_values: int = len(rhoa)

    num_cols: int = nbr_electrodes - 3
    num_lines: int = nbr_electrodes // 2 - 1

    result: np.ndarray[np.float64] = np.zeros((num_lines, num_cols), dtype=np.float64)

    value_index: int = 0
    for i in range(num_lines):
        start_col: int = i
        end_col: int = num_cols - i
        num_values_this_row: int = end_col - start_col
        result[i, start_col:end_col] = rhoa[value_index : value_index + num_values_this_row]
        value_index += num_values_this_row
    
    print(f"Time taken: {perf_counter() - start:.2f} seconds for {nbr_electrodes} electrodes.")
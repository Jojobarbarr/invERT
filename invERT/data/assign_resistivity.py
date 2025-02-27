import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from time import perf_counter
import math
import random as rd
import pygimli as pg
from tqdm import tqdm
import pygimli.physics.ert as ert


def detransform(log_res: float | np.ndarray[float]
                ) -> float | np.ndarray[float]:
    return 2 * 10 ** (4 * log_res)


def compute_active_columns(row: int,
                           is_even_row: bool,
                           total_cols: int,
                           offset: int
                           ) -> tuple[int, int]:
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


def target_section(nbr_npz: int,
                   dataset_path: Path
                   ) -> np.ndarray[np.int8]:
    # Randomly select a section from a random .npz file
    random_npz: int = rd.randint(0, nbr_npz - 1)
    with np.load(
        dataset_path / f"{random_npz}.npz",
        mmap_mode='r'
    ) as multi_array:
        section_id: int = rd.randint(0, len(multi_array["arr_0"]) - 1)
        return multi_array["arr_0"][section_id]


def define_electrodes_param() -> tuple[int, int, float]:
    nbr_electrodes: int = rd.randint(24, 96)
    # We keep 2 or 3 pixels between each electrodes
    if nbr_electrodes * 3 < section.shape[1]:
        total_pixels_to_keep: int = (nbr_electrodes - 1) * 3
    else:
        total_pixels_to_keep: int = (nbr_electrodes - 1) * 2
    pixel_length: float = rd.randint(1, 20) / 2
    return nbr_electrodes, total_pixels_to_keep, pixel_length


def generate_target(total_pixels_to_keep: int,
                    section: np.ndarray[np.int8]
                    ) -> np.ndarray[np.int8]:
    surface_level: int = rd.randint(
        0,
        section.shape[0] - total_pixels_to_keep // 2 - 1
    )
    left_border: int = rd.randint(
        0,
        section.shape[1] - total_pixels_to_keep - 1
    )

    depth_slice: slice = slice(
        surface_level,
        surface_level + total_pixels_to_keep // 2
    )
    width_slice: slice = slice(
        left_border,
        left_border + total_pixels_to_keep
    )

    return section[depth_slice, width_slice]


def generate_resistivity(target: np.ndarray[np.int8]
                         ) -> tuple[np.ndarray[np.float64],
                                    np.ndarray[np.float64]]:
    rock_classes: np.ndarray[np.int8] = np.unique(target)
    target_log_res: np.ndarray[np.float64] = np.zeros_like(target,
                                                           dtype=np.float64)
    for rock_class in rock_classes:
        random_log_resistivity: np.float64 = rd.uniform(0, 1)
        target_log_res[target == rock_class] = random_log_resistivity
    return target_log_res, detransform(target_log_res)


def generate_world(total_pixels_to_keep: int
                   ) -> pg.core.Mesh:
    x_arr: np.ndarray[np.float64] = np.linspace(
        0.,
        total_pixels_to_keep,
        total_pixels_to_keep + 1,
        dtype=np.float64
    )
    y_arr: np.ndarray[np.float64] = np.linspace(
        -(total_pixels_to_keep // 2),
        0,
        total_pixels_to_keep // 2 + 1,
        dtype=np.float64
    )
    return pg.createGrid(x=x_arr, y=y_arr, worldBoundaryMarker=True)


def generate_electrode_array(total_pixels_to_keep: int,
                             nbr_electrodes: int,
                             scheme_names: dict[str, str],
                             pixel_length: float
                             ) -> tuple[pg.DataContainerERT, str]:
    elec_array: np.float64 = np.linspace(
        0.,
        total_pixels_to_keep,
        nbr_electrodes,
        dtype=np.float64  # np.float64 to be compatible with C++ double
    )
    scheme_name = rd.choice(list(scheme_names.keys()))
    return ert.createData(elecs=elec_array,
                          schemeName=scheme_name,
                          spacing=pixel_length), scheme_name


def process_pseudo_section_wenner_array(rhoa: list[float],
                                        nbr_electrodes: int
                                        ) -> np.ndarray[np.float64]:
    num_rows: int = (nbr_electrodes - 1) // 3
    num_cols: int = nbr_electrodes - 3

    even_num_cols: bool = (num_cols % 2 == 0)
    if even_num_cols:
        # We want non even number of columns to be able to center the
        # triangle.
        num_cols += 1

    offset: int = (nbr_electrodes - 1) % 2

    pseudo_section: np.ndarray[np.float64] = np.zeros(
        (num_rows, num_cols), dtype=np.float64)
    value_index: int = 0

    for i in range(num_rows):
        # Determine if the current row is considered "even" based on num_cols
        # parity
        is_even_row = (i % 2 == 0) if even_num_cols else (i % 2 == 1)

        col_start, col_end = compute_active_columns(
            i, is_even_row, num_cols, offset)

        for j in range(col_start, col_end):
            # For even rows, use a special average at the center column
            if is_even_row and j == (num_cols - 1) // 2:
                pseudo_section[i, j] = (rhoa[value_index - 1] + rhoa[value_index]) / 2
            else:
                pseudo_section[i, j] = rhoa[value_index]
                value_index += 1
    return pseudo_section


def process_pseudo_section_schlumberger_array(rhoa: list[float],
                                              nbr_electrodes: int
                                              ) -> np.ndarray[np.float64]:
    num_cols: int = nbr_electrodes - 3
    num_lines: int = nbr_electrodes // 2 - 1

    pseudo_section: np.ndarray[np.float64] = np.zeros(
        (num_lines, num_cols), dtype=np.float64)

    value_index: int = 0
    for i in range(num_lines):
        start_col: int = i
        end_col: int = num_cols - i
        num_values_this_row: int = end_col - start_col
        pseudo_section[i, start_col:end_col] = rhoa[value_index: value_index +
                                            num_values_this_row]
        value_index += num_values_this_row

    return pseudo_section


def save_sample_hdf5(filepath, samples):
    with h5py.File(filepath, "w") as f:
        f.create_dataset("nbr_electrodes", data=[s[0] for s in samples])
        f.create_dataset("pixel_length", data=[s[1] for s in samples])
        f.create_dataset("scheme_name", data=np.array([s[2] for s in samples], dtype="S"))
        f.create_dataset("sample_log_res", data=np.array([s[3] for s in samples]), compression="lzf")
        f.create_dataset("pseudo_section", data=np.array([s[4] for s in samples]), compression="lzf")


if __name__ == "__main__":
    dataset_path: Path = Path("../../../dataset/clean_reduced_unified")
    output_path: Path = Path("../../../dataset/processed")

    random_gen: np.random.Generator = np.random.default_rng()

    nbr_npz: int = len(list(dataset_path.glob("*.npz")))

    scheme_names: dict[str, str] = {
        "wa": "Wenner array",
        "slm": "Schlumberger array"
    }

    N_SAMPLES: int = 1000
    for sample_to_process in tqdm(range(N_SAMPLES),
                                  desc="Processing samples",
                                  unit="sample",
                                  total=N_SAMPLES):
        section: np.ndarray[np.int8] = target_section(nbr_npz, dataset_path)

        nbr_electrodes, total_pixels_to_keep, pixel_length = \
            define_electrodes_param()

        target: np.ndarray = generate_target(total_pixels_to_keep, section)

        target_log_res, target_res = generate_resistivity(target)

        world: pg.core.Mesh = generate_world(total_pixels_to_keep)

        scheme, scheme_name = generate_electrode_array(
            total_pixels_to_keep,
            nbr_electrodes,
            scheme_names,
            pixel_length
        )

        result: pg.DataContainerERT = ert.simulate(
            world,
            res=target_res.ravel(),
            scheme=scheme
        )

        if scheme_name == "wa":
            pseudo_section: np.ndarray[np.float64] = \
                process_pseudo_section_wenner_array(result['rhoa'], nbr_electrodes)
        else:
            pseudo_section: np.ndarray[np.float64] = \
                process_pseudo_section_schlumberger_array(result['rhoa'], nbr_electrodes)
        
        sample: list[int, int, str, np.ndarray[np.float64], np.ndarray[np.float64]] = \
            [nbr_electrodes, pixel_length, scheme_name, target_log_res, pseudo_section]
